"""Paired, bounded RainH experiment starting from a completed LSRW task.

This is a diagnostic probe, not a replacement for a full continual benchmark.
No old training images enter the optimizer; old test images are diagnostic only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.data.datasets import build_task_loaders
from src.losses.losses import TotalLoss
from src.models.odr_lora import ODRLoRAConv2d
from src.models.restoration_net import RestorationNet
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.inference import predict_image
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.raw_gpm import fit_raw_gpm_banks, install_gpm_bank


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def selected_indices(count, limit):
    return np.linspace(0, count - 1, min(count, limit), dtype=int).tolist()


def summarize_rows(rows):
    return {name: dict(
        num=len(values),
        **{metric: float(np.mean([row[metric] for row in values]))
           for metric in ('psnr', 'ssim', 'lpips')})
        for name, values in rows.items()}


@torch.no_grad()
def evaluate(model, datasets, indices, device, lpips_fn, tile_size):
    model.eval()
    rows = {}
    for name, dataset in datasets.items():
        rows[name] = []
        for index in indices[name]:
            sample = dataset[index]
            x = sample['input'].unsqueeze(0).to(device)
            gt = sample['gt'].unsqueeze(0).to(device)
            prediction = predict_image(model, x, 1, tile_size, 64)
            rows[name].append(dict(
                index=index, psnr=compute_psnr(prediction, gt),
                ssim=compute_ssim(prediction, gt),
                lpips=float(lpips_fn(prediction * 2 - 1, gt * 2 - 1).mean())))
        print(f'EVAL {name}: {summarize_rows({name: rows[name]})[name]}', flush=True)
    return rows


@torch.no_grad()
def update_diagnostics(model):
    rows = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, ODRLoRAConv2d):
            continue
        M = layer.dual_gpms['global'].basis
        B, A = layer.task_B[1], layer.task_A[1]
        delta = A @ B
        bnorm, dnorm = float(B.norm()), float(delta.norm())
        row_norms = B.norm(dim=1)
        active = row_norms > 1e-5
        active_B = B[active]
        gram_error = (float((active_B @ active_B.T - torch.eye(
            active_B.shape[0], device=B.device)).norm()) / max(1, active_B.shape[0]) ** 0.5)
        rows[name] = dict(
            dim=layer.d, gpm_rank=0 if M is None else M.shape[1],
            at_cap=M is not None and M.shape[1] == layer.dual_gpms['global'].max_basis,
            basis_norm=bnorm, update_norm=dnorm,
            basis_overlap=0.0 if M is None else float((B @ M).norm()) / max(bnorm, 1e-12),
            update_overlap=0.0 if M is None else float((delta @ M).norm()) / max(dnorm, 1e-12),
            effective_rank=int(torch.linalg.matrix_rank(B.float(), atol=1e-5, rtol=0)),
            allocated_rows=B.shape[0],
            out_channels=layer.out_channels,
            shared_active_rows=int(active[:layer.share_rank].sum()),
            private_active_rows=int(active[layer.share_rank:].sum()),
            active_row_norm_mean=float(row_norms[active].mean()) if active.any() else 0.0,
            active_gram_error=gram_error,
        )
    return dict(
        basis_overlap_mean=float(np.mean([v['basis_overlap'] for v in rows.values()])),
        update_overlap_mean=float(np.mean([v['update_overlap'] for v in rows.values()])),
        update_overlap_max=max(v['update_overlap'] for v in rows.values()),
        zero_rank_layers=sum(v['effective_rank'] == 0 for v in rows.values()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        active_trainable_parameters=sum(v['out_channels'] * (
            v['shared_active_rows'] + v['private_active_rows']) for v in rows.values()),
        layers=rows,
    )


@torch.no_grad()
def coverage(model, dataset, device, count=8, patch_size=128):
    """Measure OLD GPM energy coverage on actual convolution input patches.

    Use held-out center crops and up to 256 evenly spaced columns per layer.
    These are diagnostic sample estimates, not a claim of population coverage.
    """
    totals, hooks = {}, []
    model.eval()

    def make_hook(name):
        def hook(layer, args):
            x = args[0]
            H = F.unfold(x, layer.kernel_size, padding=layer.padding,
                         stride=layer.stride)
            H = H.permute(1, 0, 2).reshape(layer.d, -1).float()
            columns = torch.linspace(0, H.shape[1] - 1, min(256, H.shape[1]),
                                     device=H.device).long()
            H = H[:, columns]
            M = layer.dual_gpms['global'].basis
            total = float(H.square().sum())
            covered = 0.0 if M is None else float((M.T @ H).square().sum())
            row = totals.setdefault(name, dict(dim=layer.d,
                rank=0 if M is None else M.shape[1], total_energy=0.0,
                covered_energy=0.0, samples=0))
            row['total_energy'] += total
            row['covered_energy'] += covered
            row['samples'] += H.shape[1]
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, ODRLoRAConv2d):
            hooks.append(layer.register_forward_pre_hook(make_hook(name)))
    try:
        for index in selected_indices(len(dataset), count):
            x = dataset[index]['input']
            h, w = x.shape[-2:]
            top, left = max(0, (h-patch_size)//2), max(0, (w-patch_size)//2)
            model(x[:, top:top+patch_size, left:left+patch_size].unsqueeze(0).to(device), 1)
    finally:
        for hook in hooks:
            hook.remove()
    for row in totals.values():
        row['coverage'] = row['covered_energy'] / max(row['total_energy'], 1e-12)
    values = [v['coverage'] for v in totals.values()]
    return dict(mean=float(np.mean(values)), min=min(values), max=max(values),
                below_90=sum(v < 0.9 for v in values), layers=totals)


def plot_results(output, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm, curve in curves.items():
        steps = [r['steps'] for r in curve]
        for ax, task in zip(axes[:2], ('lowlight', 'rainh')):
            ax.plot(steps, [r['metrics'][task]['psnr'] for r in curve],
                    marker='o', label=arm)
            ax.set_title(f'{task}: fixed-subset PSNR')
            ax.set_ylabel('dB')
        axes[2].plot(steps, [r['diagnostics']['update_overlap_mean'] for r in curve],
                     marker='o', label=arm)
    axes[2].set_title('Mean update overlap with old GPM')
    old_values = [r['metrics']['lowlight']['psnr'] for curve in curves.values() for r in curve]
    axes[0].set_ylim(np.floor(min(old_values)) - 0.5, np.ceil(max(old_values)) + 0.5)
    max_overlap = max(r['diagnostics']['update_overlap_mean']
                      for curve in curves.values() for r in curve)
    axes[2].set_ylim(0, max(1e-6, 1.1 * max_overlap))
    if max_overlap < 1e-6:
        axes[2].set_title('Update overlap (all < 1e-6)')
    for ax in axes:
        ax.set_xlabel('Optimizer steps')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


def write_report(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    coverage_old = json.loads((output / 'coverage_lowlight.json').read_text())
    curves = {arm: json.loads((output / arm / 'curve.json').read_text())
              for arm in ('baseline', 'protected')}
    initial, arms = summary['initial'], summary['arms']
    lines = ['# LSRW → RainH 短程对照结果', '',
             f'本表使用完整 LSRW {initial["lowlight"]["num"]} 张、'
             f'RainH {initial["rainh"]["num"]} 张测试图像。', '',
             '| 组别 | LSRW PSNR | LSRW SSIM | RainH PSNR | RainH SSIM |',
             '|---|---:|---:|---:|---:|']
    metrics = {'初始 checkpoint': initial,
               '原实现': arms['baseline']['metrics'],
               '完整更新保护': arms['protected']['metrics']}
    for label, values in metrics.items():
        old, new = values['lowlight'], values['rainh']
        lines.append(f"| {label} | {old['psnr']:.4f} | {old['ssim']:.4f} | "
                     f"{new['psnr']:.4f} | {new['ssim']:.4f} |")
    steps = manifest['arguments']['rounds'] * manifest['arguments']['steps_per_round']
    lines += ['', f'每组 {steps} 次优化器更新，batch_size={manifest["arguments"]["batch_size"]}，'
              '相同的裁剪、增强、校准数据、随机种子、固定学习率及 L1/LPIPS 损失。',
              '旧任务测试图像仅用于诊断，未进入优化器，也未用于早停或挑选模型。', '']
    for arm, label in [('baseline', '原实现'), ('protected', '保护组')]:
        result = arms[arm]
        leakage = curves[arm][-1]['diagnostics']['update_overlap_mean']
        lines.append(f'- {label}：LSRW PSNR 变化 {result["old_psnr_change"]:+.4f} dB；'
                     f'末轮更新泄漏逐层均值 {leakage:.6g}；'
                     f'合并检查 PSNR 变化 {result["merge_psnr_difference"]:.6f} dB。')
    lines += ['', '泄漏定义为 ||ΔW M_old||F / ||ΔW||F，再逐层平均；不是遗忘百分比。', '',
              f'旧 GPM 在抽样 LSRW 真实卷积输入上的逐层平均能量覆盖率为 '
              f'{coverage_old["mean"]:.2%}，最低为 {coverage_old["min"]:.2%}，'
              f'{coverage_old["below_90"]}/{len(coverage_old["layers"])} 层低于 90%。'
              '该诊断只使用 8 张固定中心裁剪、每层每图最多 256 个 patch，不能等同于全数据覆盖率。', '',
              '## 如何解释', '',
              '比较旧任务保留效果时，也须同时比较 RainH 指标。保护旧子空间可能减缓新任务学习；'
              '不能把较小遗忘直接解释为兼顾全部任务的最终解决方案。',
              '本实验固定 GPM 容量与旧基，仅检验完整更新投影的影响。'
              '后续应分别研究真实激活统计、GPM 容量/压缩与多任务长程表现。',
              '这是单种子、短程筛查；曲线为固定子集（LSRW 20、RainH 30），'
              '上表为完整测试集，二者不能混用。', '',
              f'![固定子集曲线]({(output / "curves.png").resolve()})', '',
              '## 文件', '',
              '- summary.json：完整起点/终点指标、合并误差。',
              '- baseline/curve.json、protected/curve.json：每轮新旧任务指标及逐层诊断。',
              '- coverage_lowlight.json、coverage_rainh.json：逐层 GPM 覆盖估计。',
              '- manifest.json：原配置、参数、随机种子、测试索引与 checkpoint SHA256。',
              '- probe_*.pth：诊断模型，未更新 GPM，不用于继续正式任务。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def write_gpm_report(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    lines = ['# GPM 统计与容量短程对照', '',
        '三组均从同一 LSRW 最终 checkpoint 开始，并启用完整更新保护。',
        'raw64/raw128 从固定的旧任务训练样本重建保护空间，完整保留旧方向，'
        '增加真实卷积输入的残差 PCA 方向；未使用旧测试集建模。', '',
        '| 方案 | LSRW PSNR | RainH PSNR | 两任务等权平均 | LSRW SSIM | RainH SSIM | 原始激活覆盖估计 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    initial = summary['initial']
    lines.append(f'| 训练前 | {initial["lowlight"]["psnr"]:.4f} | '
        f'{initial["rainh"]["psnr"]:.4f} | '
        f'{(initial["lowlight"]["psnr"]+initial["rainh"]["psnr"])/2:.4f} | '
        f'{initial["lowlight"]["ssim"]:.4f} | {initial["rainh"]["ssim"]:.4f} | — |')
    averages = {}
    for arm, result in summary['arms'].items():
        old, new = result['metrics']['lowlight'], result['metrics']['rainh']
        cov = json.loads((output / arm / 'coverage_lowlight.json').read_text())
        averages[arm] = (old['psnr']+new['psnr'])/2
        lines.append(f'| {arm} | {old["psnr"]:.4f} | {new["psnr"]:.4f} | '
            f'{averages[arm]:.4f} | {old["ssim"]:.4f} | {new["ssim"]:.4f} | {cov["mean"]:.2%} |')
    best = max(averages, key=averages.get)
    lines += ['', f'本次短程完整测试集平均 PSNR 最高：{best}（{averages[best]:.4f} dB）。',
        '这只描述本次单种子筛查，不是根据测试集选出的正式最终模型。', '',
        f'每组 {manifest["arguments"]["rounds"]} 轮 × '
        f'{manifest["arguments"]["steps_per_round"]} 步，batch='
        f'{manifest["arguments"]["batch_size"]}；相同新任务数据、优化器、固定学习率及 L1/LPIPS 损失。',
        f'原始 GPM 校准使用 {len(manifest["old_calibration_indices"])} 张旧训练图的随机裁剪，'
        '固定 seed=777。旧图仅用于记忆统计，不进入 RainH 的优化器。', '',
        '| 方案 | 两任务平均 SSIM ↑ | 两任务平均 LPIPS ↓ |',
        '|---|---:|---:|']
    for arm, result in summary['arms'].items():
        values = list(result['metrics'].values())
        lines.append(f'| {arm} | {np.mean([v["ssim"] for v in values]):.4f} | '
                     f'{np.mean([v["lpips"] for v in values]):.4f} |')
    lines += ['',
        '## 逐组诊断', '']
    for arm in summary['arms']:
        curves = json.loads((output / arm / 'curve.json').read_text())
        diag = curves[-1]['diagnostics']
        ranks = [v['gpm_rank'] for v in diag['layers'].values()]
        lines.append(f'- {arm}：实际 GPM 秩均值 {np.mean(ranks):.2f}、最大 {max(ranks)}；'
            f'没有有效更新方向的层 {diag["zero_rank_layers"]}；'
            f'平均更新泄漏 {diag["update_overlap_mean"]:.6g}。')
    lines += ['', '## 解释边界', '',
        '覆盖率来自独立旧测试图的固定中心裁剪，是逐层非加权均值；不代表整体模型覆盖。'
        '测试图未用于 GPM 拟合。正式终点指标使用完整 LSRW 50 张、RainH 100 张，'
        '曲线使用固定的 20/30 张子集。',
        'current32→raw64 同时改变统计和容量，并增加了旧训练样本的离线校准访问；'
        '不能把差异归因于容量一个因素。raw64→raw128 使用同一份校准协方差，'
        '且保留相同旧方向，主要比较容量上限。',
        '64/128 是容量上限，不要求每层填满；若已覆盖校准样本 99% 能量则停止添加。'
        '覆盖更高可能让新任务可学习空间更小，不保证整体 PSNR 更高。',
        '本实验没有改损失、LoRA 秩、网络深度，也没有继续正式三任务训练。'
        '诊断 checkpoint 未更新任务末的 GPM，不用于正式后续任务。', '',
        f'![曲线]({(output / "curves.png").resolve()})', '',
        '原始数据：summary.json、各组 curve.json / coverage_*.json、raw_fit_*.json、manifest.json。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def write_basis_report(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    lines = ['# 投影后有效基与私有秩短程对照', '',
        '三组均从同一 LSRW checkpoint 开始，固定旧 GPM-32、完整更新保护、'
        '网络、损失和优化器。没有重新访问旧训练集拟合记忆。', '',
        '| 方案 | LSRW PSNR | RainH PSNR | 两任务平均 PSNR | 平均 SSIM ↑ | 平均 LPIPS ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    averages = {}
    for arm, result in summary['arms'].items():
        values = result['metrics']
        average = (values['lowlight']['psnr'] + values['rainh']['psnr']) / 2
        averages[arm] = average
        lines.append(f'| {arm} | {values["lowlight"]["psnr"]:.4f} | '
            f'{values["rainh"]["psnr"]:.4f} | {average:.4f} | '
            f'{np.mean([v["ssim"] for v in values.values()]):.4f} | '
            f'{np.mean([v["lpips"] for v in values.values()]):.4f} |')
    lines += ['', f'本轮平均 PSNR 最高：{max(averages, key=averages.get)}。', '',
        f'每组 {manifest["arguments"]["rounds"]} 轮 × '
        f'{manifest["arguments"]["steps_per_round"]} 步，batch='
        f'{manifest["arguments"]["batch_size"]}；不根据旧测试集早停或挑选 checkpoint。',
        '终点使用完整 LSRW 50 张、RainH 100 张测试集，曲线为固定 20/30 张子集。', '',
        '## 三组差异', '',
        '- current_basis：原有软投影/SVD，最后对整个 B 做严格投影。',
        '- effective_r8：先严格投影共享基及新特征，剔除近零奇异方向，'
        '归一化有效共享方向；在其正交补中提取单位私有方向。私有基础秩仍为 8。',
        '- effective_r16：使用相同新构造，仅把新任务私有基础秩从 8 增到 16；'
        '按层尺度增长，上限仍为 32，GPM 上限保持 32。旧任务 adapter 不扩展。', '',
        '## 初始校准后的有效容量', '',
        '| 方案 | 分配的 A 参数数 | 对应非零基行的 A 参数数 | 每层平均有效秩 | 非零基行平均范数 |',
        '|---|---:|---:|---:|---:|']
    for arm in summary['arms']:
        curve = json.loads((output / arm / 'curve.json').read_text())
        diag = curve[0]['diagnostics']
        rows = list(diag['layers'].values())
        lines.append(f'| {arm} | {diag["trainable_parameters"]} | '
            f'{diag["active_trainable_parameters"]} | '
            f'{np.mean([v["effective_rank"] for v in rows]):.2f} | '
            f'{np.mean([v["active_row_norm_mean"] for v in rows]):.4f} |')
    lines += ['', '## 保护及合并检查', '']
    for arm, result in summary['arms'].items():
        curve = json.loads((output / arm / 'curve.json').read_text())
        diag = curve[-1]['diagnostics']
        lines.append(f'- {arm}：平均更新泄漏 {diag["update_overlap_mean"]:.6g}；'
            f'合并检查 PSNR 变化 {result["merge_psnr_difference"]:.6f} dB。')
    lines += ['', '## 解释边界', '',
        'current_basis 与 effective_r8 的分配参数预算相同，但有效秩、基范数与方向选择可能不同；'
        '不能把差异仅解释为一种归一化效应。effective_r16 分配更多 A 参数，'
        '因此是增容实验，不是等参数比较。',
        '有效秩不足时保持零行，不用随机方向填满。共享和私有的有效方向互相正交，'
        '且始终在旧 GPM 正交补内。新方向也沿用空间/频带采样，'
        '本轮没有改变记忆统计方式或保护空间。',
        '这是单种子、短程筛查，不能证明长程训练或更多任务的最终效果；'
        '诊断模型未更新任务末 GPM，不作为正式后续任务起点。', '',
        f'![曲线]({(output / "curves.png").resolve()})', '',
        '原始材料：summary.json、各组 curve.json/arm_config.yaml、manifest.json、probe_*.pth。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lsrw_rainh.yaml')
    parser.add_argument('--checkpoint', default='outputs/lsrw/task_0_final.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--steps-per-round', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--memory-fraction', type=float, default=0.32)
    parser.add_argument('--tile-size', type=int, default=0)
    parser.add_argument('--resume-experiment', action='store_true',
                        help='Resume saved rounds in the same probe directory')
    parser.add_argument('--study', choices=('protection', 'gpm', 'basis'), default='protection')
    parser.add_argument('--gpm-calibration-samples', type=int, default=32)
    args = parser.parse_args()
    if min(args.rounds, args.steps_per_round, args.batch_size, args.gpm_calibration_samples) < 1:
        parser.error('rounds, steps-per-round and batch-size must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume_experiment)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if [t['name'] for t in cfg['data']['tasks']] != ['lowlight', 'rainh']:
        raise ValueError('Probe requires exactly lowlight -> rainh task order')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this bounded experiment')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    device = torch.device('cuda:0')
    cfg['experiment']['device'] = 'cuda:0'
    seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    datasets = {name: build_task_loaders(cfg, name)[1].dataset
                for name in ('lowlight', 'rainh')}
    train_dataset = build_task_loaders(cfg, 'rainh')[0].dataset
    subset = {name: selected_indices(len(ds), 20 if name == 'lowlight' else 30)
              for name, ds in datasets.items()}
    full = {name: list(range(len(ds))) for name, ds in datasets.items()}
    arm_specs = {
        'protection': [('baseline', False, None), ('protected', True, None)],
        'gpm': [('current32', True, None), ('raw64', True, 64), ('raw128', True, 128)],
        'basis': [('current_basis', True, None), ('effective_r8', True, None),
                  ('effective_r16', True, None)],
    }[args.study]
    reference_arm = arm_specs[0][0]
    old_train = build_task_loaders(cfg, 'lowlight')[0].dataset if args.study == 'gpm' else None
    old_indices = (selected_indices(len(old_train), args.gpm_calibration_samples)
                   if old_train is not None else [])
    criterion = TotalLoss(cfg).to(device).eval()
    if not criterion.perceptual.use_lpips:
        raise RuntimeError('LPIPS must be available; no silent L1 fallback in this probe')
    criterion.requires_grad_(False)
    curves, results, baseline_rows = {}, {}, None
    initial_metrics = None
    if args.resume_experiment:
        initial_path = output / 'initial_full_metrics.json'
        if initial_path.exists():
            initial_metrics = json.loads(initial_path.read_text())
    manifest = dict(arguments=vars(args), config=cfg, seed=42,
        subset_indices=subset, train_samples=len(train_dataset),
        test_samples={k: len(v) for k, v in datasets.items()},
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        old_data_usage=('fixed old TRAIN subset for memory calibration only; tests diagnostic only'
                        if args.study == 'gpm' else 'test diagnostics only; never used by optimizer'),
        old_calibration_indices=old_indices,
        old_calibration_paths=([old_train.input_paths[i] for i in old_indices]
                               if old_train is not None else []),
        arm_specs=arm_specs,
        basis_protocol=('GPM-32 fixed; legacy vs projected-first unit bases at rank base 8; '
                        'then increase ONLY the new task private rank base to 16'
                        if args.study == 'basis' else 'unchanged'),
        calibration_protocol='seed 777; uncentered raw convolution patches; preserve all old directions',
        scheduler='constant LR matched between arms; not the 400-epoch schedule',
        metric_protocol='full image unless tile-size specified; existing PSNR/SSIM clamp',
        projection='entire frozen B projected once against OLD GPM after calibration')
    manifest_path = output / 'manifest.json'
    if args.resume_experiment:
        original = json.loads(manifest_path.read_text())
        for key in ('rounds', 'steps_per_round', 'batch_size', 'tile_size', 'config', 'checkpoint'):
            if original['arguments'][key] != vars(args)[key]:
                raise ValueError(f'Resume changed experiment argument: {key}')
        for key in ('study', 'gpm_calibration_samples'):
            if original['arguments'].get(key, 'protection' if key == 'study' else 32) != vars(args)[key]:
                raise ValueError(f'Resume changed experiment argument: {key}')
        if (original['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or original['config'] != cfg):
            raise ValueError('Resume changed source checkpoint or configuration')
    else:
        write_json(manifest_path, manifest)
    start = time.monotonic()
    for arm, protected, capacity in arm_specs:
        seed_all(42)
        arm_dir = output / arm
        arm_dir.mkdir(exist_ok=args.resume_experiment)
        if args.resume_experiment and (arm_dir / 'final_full_metrics.json').exists():
            results[arm] = json.loads((arm_dir / 'final_full_metrics.json').read_text())
            curves[arm] = json.loads((arm_dir / 'curve.json').read_text())
            continue
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg['odr_lora']['full_update_protection'] = protected
        if capacity is not None:
            arm_cfg['odr_lora']['gpm_max_basis'] = capacity
            arm_cfg['odr_lora']['gpm_statistics'] = 'raw'
        if args.study == 'basis':
            arm_cfg['odr_lora']['gpm_max_basis'] = 32
            arm_cfg['odr_lora']['gpm_statistics'] = 'frequency'
            arm_cfg['odr_lora']['basis_construction'] = (
                'legacy' if arm == 'current_basis' else 'projected_svd')
            arm_cfg['model']['new_task_private_rank_base'] = 16 if arm == 'effective_r16' else 8
        config_path = arm_dir / 'arm_config.yaml'
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(arm_cfg, sort_keys=False))
        model = RestorationNet(arm_cfg).to(device)
        _, task_idx, state = load_checkpoint(model, args.checkpoint,
                                            map_location=device, return_state=True)
        model.load_state_dict(state['model'], strict=True)
        if task_idx != 0 or state.get('current_task_idx') != -1 or state.get('num_tasks') != 1:
            raise ValueError('Expected completed task_0_final with exactly one task')
        if any(layer.dual_gpms['global'].basis is None for layer in model.modules()
               if isinstance(layer, ODRLoRAConv2d)):
            raise ValueError('Checkpoint lacks protected GPM; do not use merged.pth')
        if args.study == 'gpm':
            if not all((output / f'raw_bank_{cap}.pth').exists() for cap in (64, 128)):
                seed_all(777)
                banks = fit_raw_gpm_banks(model, old_train, old_indices, device,
                                          batch_size=args.batch_size)
                for cap, bank in banks.items():
                    temporary = output / f'raw_bank_{cap}.pth.tmp'
                    torch.save(bank, temporary)
                    temporary.replace(output / f'raw_bank_{cap}.pth')
                    write_json(output / f'raw_fit_{cap}.json', bank['diagnostics'])
                del banks
            if capacity is not None:
                bank = torch.load(output / f'raw_bank_{capacity}.pth',
                                  map_location='cpu', weights_only=False)
                install_gpm_bank(model, bank, capacity)
                del bank
            for name, ds in datasets.items():
                cover_path = arm_dir / f'coverage_{name}.json'
                if not cover_path.exists():
                    write_json(cover_path, coverage(model, ds, device))
        model.add_task()
        model.set_current_task(1)
        similarity = state.get('task_freq_energy', {})
        from src.models.freq_decomp import multilevel_dwt, freq_band_energy_from_bands
        seed_all(101)
        calibration = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0)
        ratios = []
        for i, batch in enumerate(calibration):
            ratios.append(freq_band_energy_from_bands(multilevel_dwt(batch['input'], 2)))
            if i == 4:
                break
        a = np.mean(ratios, axis=0)
        b = np.asarray(similarity['lowlight'])
        sim = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        seed_all(202)
        for i, batch in enumerate(DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0)):
            model.calibrate(batch['input'].to(device), 1, sim)
            if i == 5:
                break
        model.freeze_base_and_old(1)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg['train']['lr'],
                                     weight_decay=cfg['train']['weight_decay'])
        if arm == reference_arm and initial_metrics is None:
            print('MEASURING original GPM coverage on held-out raw activations', flush=True)
            for name, ds in datasets.items():
                write_json(output / f'coverage_{name}.json', coverage(model, ds, device))
            baseline_rows = evaluate(model, datasets, full, device,
                                     criterion.perceptual.lpips, args.tile_size)
            initial_metrics = summarize_rows(baseline_rows)
            write_json(output / 'initial_full_metrics.json', initial_metrics)
        start_round = 1
        if args.resume_experiment and (arm_dir / 'probe_latest.pth').exists():
            curves[arm] = json.loads((arm_dir / 'curve.json').read_text())
            epoch, _, saved = load_checkpoint(model, str(arm_dir / 'probe_latest.pth'),
                optimizer=optimizer, map_location=device, return_state=True)
            if len(curves[arm]) != epoch + 1 or saved['full_update_protection'] != protected:
                raise ValueError('Probe checkpoint and curve disagree')
            start_round = epoch + 1
        else:
            if baseline_rows is not None:
                initial_subset = {name: [row for row in values if row['index'] in subset[name]]
                                  for name, values in baseline_rows.items()}
                initial_subset_metrics = summarize_rows(initial_subset)
            else:
                initial_subset_metrics = json.loads((output / reference_arm / 'curve.json').read_text())[0]['metrics']
            curves[arm] = [dict(round=0, steps=0, metrics=initial_subset_metrics,
                                diagnostics=update_diagnostics(model))]
        write_json(arm_dir / 'curve.json', curves[arm])
        for round_idx in range(start_round, args.rounds + 1):
            seed_all(1000 + round_idx)
            model.train()
            losses = []
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0)
            if args.steps_per_round > len(loader):
                raise ValueError('steps-per-round exceeds one training pass')
            for step, batch in enumerate(loader, 1):
                x, gt = batch['input'].to(device), batch['gt'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x, 1), gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg['train']['grad_clip'])
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 10 == 0:
                    print(f'{arm} round {round_idx}/{args.rounds} step {step}: loss={losses[-1]:.5f}', flush=True)
                if step >= args.steps_per_round:
                    break
            rows = evaluate(model, datasets, subset, device,
                            criterion.perceptual.lpips, args.tile_size)
            row = dict(round=round_idx, steps=round_idx*args.steps_per_round,
                       loss=float(np.mean(losses)), metrics=summarize_rows(rows),
                       diagnostics=update_diagnostics(model))
            curves[arm].append(row)
            write_json(arm_dir / 'curve.json', curves[arm])
            save_checkpoint(model, optimizer, round_idx, 1, str(arm_dir / 'probe_latest.pth'),
                extra={'probe_only': True, 'in_progress': True, 'steps': row['steps'],
                       'full_update_protection': protected,
                       'task_freq_energy': similarity, 'task_similarity': {1: sim}})
            write_json(output / 'status.json', dict(arm=arm, round=round_idx,
                       elapsed_seconds=time.monotonic()-start))
            print(f'{arm} round {round_idx} metrics={row["metrics"]} overlap={row["diagnostics"]["update_overlap_mean"]:.6g}', flush=True)
        full_cache = arm_dir / 'final_unmerged_metrics.json'
        if args.resume_experiment and full_cache.exists():
            full_metrics = json.loads(full_cache.read_text())
        else:
            final_rows = evaluate(model, datasets, full, device,
                                  criterion.perceptual.lpips, args.tile_size)
            full_metrics = summarize_rows(final_rows)
            write_json(full_cache, full_metrics)
        # Verify the learned function survives merging before exporting a probe model.
        sample = datasets['lowlight'][0]['input'][:, :128, :128].unsqueeze(0).to(device)
        with torch.no_grad():
            before = model(sample, 1)
            model.merge_task(1)
            after = model(sample, -1)
        merge_error = float((before-after).abs().max())
        merge_mean_error = float((before-after).abs().mean())
        target = datasets['lowlight'][0]['gt'][:, :128, :128].unsqueeze(0).to(device)
        merge_psnr_difference = abs(compute_psnr(before, target)-compute_psnr(after, target))
        # Reordering two convolutions into one accumulates FP32 rounding over
        # 162 layers. Check image quality as well as bounded pixel differences.
        if merge_error > 1e-3 or merge_mean_error > 1e-4 or merge_psnr_difference > 0.01:
            raise RuntimeError(f'Merge error max={merge_error}, mean={merge_mean_error}, '
                               f'PSNR change={merge_psnr_difference} dB')
        results[arm] = dict(metrics=full_metrics, similarity=sim, merge_max_error=merge_error,
            merge_mean_error=merge_mean_error, merge_psnr_difference=merge_psnr_difference,
            old_psnr_change=full_metrics['lowlight']['psnr']-initial_metrics['lowlight']['psnr'],
            new_psnr_change=full_metrics['rainh']['psnr']-initial_metrics['rainh']['psnr'])
        save_checkpoint(model, None, args.rounds, 1, str(arm_dir / 'probe_merged.pth'),
                        extra={'probe_only': True, 'gpm_not_updated': True})
        write_json(arm_dir / 'final_full_metrics.json', results[arm])
        write_json(output / 'summary.json', dict(initial=initial_metrics, arms=results))
        del optimizer, params, model, state
        torch.cuda.empty_cache()
    plot_results(output, curves)
    if args.study == 'protection':
        write_report(output)
    elif args.study == 'gpm':
        write_gpm_report(output)
    else:
        write_basis_report(output)
    write_json(output / 'status.json', dict(status='complete', elapsed_seconds=time.monotonic()-start))
    print('COMPLETE', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
