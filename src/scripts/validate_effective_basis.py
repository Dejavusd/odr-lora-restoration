"""Grouped holdout, paired-seed and longer-budget LSRW -> RainH validation.

Prepare once, run arms on separate GPUs if desired, then aggregate. Test data
are retention diagnostics, never checkpoint-selection data. Outputs are probes,
not task-final checkpoints: the GPM is deliberately held fixed.
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
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.datasets import RestorationDataset, pair_images
from src.losses.losses import TotalLoss
from src.models.restoration_net import RestorationNet
from src.models.freq_decomp import multilevel_dwt, freq_band_energy_from_bands
from src.scripts.run_forgetting_probe import (
    evaluate, seed_all, selected_indices, summarize_rows, update_diagnostics, write_json,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import compute_psnr


ARMS = ('current_basis', 'effective_r16')


def file_sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def image_record(input_path, gt_path):
    with Image.open(input_path) as inp, Image.open(gt_path) as gt:
        if inp.size != gt.size:
            raise ValueError(f'Input/GT size mismatch: {input_path}, {gt_path}')
        digest = hashlib.sha256(str(gt.size).encode() + gt.convert('RGB').tobytes()).hexdigest()
        size = list(gt.size)
    return dict(input=input_path, gt=gt_path, gt_pixel_sha256=digest, size=size,
                input_sha256=file_sha(input_path), gt_sha256=file_sha(gt_path),
                file_stats=[list((os.stat(p).st_size, os.stat(p).st_mtime_ns))
                            for p in (input_path, gt_path)])


def grouped_split(records, seed=20261029, val_fraction=0.1):
    groups = sorted({row['gt_pixel_sha256'] for row in records})
    if len(groups) < 2:
        raise ValueError('Need at least two distinct clean-target groups')
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups)-1, round(len(groups)*val_fraction)))
    held_out = set(groups[:count])
    train = [i for i, row in enumerate(records) if row['gt_pixel_sha256'] not in held_out]
    val = [i for i, row in enumerate(records) if row['gt_pixel_sha256'] in held_out]
    assert set(records[i]['gt_pixel_sha256'] for i in train).isdisjoint(
        records[i]['gt_pixel_sha256'] for i in val)
    return train, val


def prepare(args, cfg, output):
    if output.exists():
        raise FileExistsError(f'Refusing to replace validation directory: {output}')
    records = {}
    for task, dataset in [('lowlight', 'LSRW'), ('rainh', 'RainH')]:
        for split in ('train', 'test'):
            root = str(Path(cfg['data']['root']) / dataset / split)
            inputs, targets = pair_images(root, root)
            if task == 'rainh' or split == 'test':
                rows = [image_record(x, y) for x, y in zip(inputs, targets)]
                records[f'{task}_{split}'] = rows
            if task == 'lowlight' and split == 'train':
                old_pair_audit = dict(pairs=len(inputs),
                    exact_filename_pairs=sum(Path(x).name == Path(y).name
                                             for x, y in zip(inputs, targets)),
                    prior_numeric_cartesian_pairs=10500,
                    warning='Historical task-0 checkpoint may have used erroneous numeric pairing; '
                            'its weights are fixed here, not retrained.')
    train, val = grouped_split(records['rainh_train'])
    train_groups = {r['gt_pixel_sha256'] for r in records['rainh_train']}
    test_groups = {r['gt_pixel_sha256'] for r in records['rainh_test']}
    if train_groups & test_groups:
        raise ValueError('RainH train/test have identical clean targets')
    manifest = dict(config=cfg, source_checkpoint=str(Path(args.checkpoint).resolve()),
        source_sha256=file_sha(args.checkpoint), seeds=args.seeds, arms=list(ARMS),
        steps=args.steps, interval=args.interval, batch_size=args.batch_size,
        split_seed=20261029, train_indices=train, validation_indices=val, records=records,
        rainh_train_groups=len(train_groups), rainh_test_groups=len(test_groups),
        validation_groups=len({records['rainh_train'][i]['gt_pixel_sha256'] for i in val}),
        old_pair_audit=old_pair_audit,
        selection='Highest RainH grouped TRAIN-holdout full-image PSNR; ties choose earlier step',
        old_test_usage='10 fixed old-test images for retention curves only; never model selection',
        test_usage='Full existing tests at final and validation-selected checkpoints only; '
                   'these tests were already consulted in previous development, not pristine confirmation',
        schedule='Constant lr=2e-4 (matched arms), L1 + 0.1 LPIPS, GPM-32 frequency fixed',
        seed_scope='Incremental-stage seeds only; ALL runs share the same task-0 pretrained checkpoint',
        longer_budget='1000 updates is 5x the prior 200-step probe, NOT full 400-epoch training',
        validation_kind='Exact decoded-clean-target grouping, not guaranteed scene-disjoint for near duplicates')
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', manifest)
    print(f'PREPARED train={len(train)} val={len(val)} groups='
          f'{len(train_groups)}/{manifest["validation_groups"]}; no exact GT overlap with test', flush=True)


def arm_config(cfg, arm):
    cfg = copy.deepcopy(cfg)
    cfg['experiment']['device'] = 'cuda:0'
    cfg['odr_lora'].update(full_update_protection=True, gpm_max_basis=32,
                          gpm_statistics='frequency',
                          basis_construction='legacy' if arm == 'current_basis' else 'projected_svd')
    cfg['model']['new_task_private_rank_base'] = 8 if arm == 'current_basis' else 16
    return cfg


def make_model(cfg, checkpoint, device):
    model = RestorationNet(cfg).to(device)
    _, idx, state = load_checkpoint(model, checkpoint, map_location=device, return_state=True)
    model.load_state_dict(state['model'], strict=True)
    return model, state


def optimizer_for(model, cfg):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_state(path, model, optimizer, step, curve, best_step, best_psnr):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 epoch=step, task_idx=1, num_tasks=2, current_task_idx=1,
                 steps=step, curve=curve, best_step=best_step, best_psnr=best_psnr,
                 rng=rng_state(), probe_only=True, gpm_not_updated=True)
    temporary = Path(str(path) + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_state(model, optimizer, path, device):
    _, _, saved = load_checkpoint(model, str(path), optimizer=optimizer,
                                  map_location=device, return_state=True)
    model.load_state_dict(saved['model'], strict=True)
    if optimizer is not None:
        # Unlike the legacy utility's compatibility fallback, validation must fail
        # loudly if optimizer state cannot be restored.
        optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return saved


def train_step(model, optimizer, criterion, batch, cfg, device):
    model.train()
    x, gt = batch['input'].to(device), batch['gt'].to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x, 1), gt)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  cfg['train']['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def psnr_only(model, dataset, indices, device):
    model.eval()
    values = []
    for index in indices:
        item = dataset[index]
        pred = model(item['input'].unsqueeze(0).to(device), 1)
        values.append(compute_psnr(pred, item['gt'].unsqueeze(0).to(device)))
    return float(np.mean(values))


def check_resume(model, optimizer, criterion, cfg, path, batch, device):
    """Replay a real optimizer step from disk, then restore the main run exactly."""
    expected_rng = rng_state()
    reference_loss = train_step(model, optimizer, criterion, batch, cfg, device)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restored, _ = make_model(cfg, str(path), device)
    restored.freeze_base_and_old(1)
    restored.set_current_task(1)
    opt = optimizer_for(restored, cfg)
    restore_state(restored, opt, path, device)
    restore_rng(expected_rng)
    replay_loss = train_step(restored, opt, criterion, batch, cfg, device)
    tensor_error = max(float((v.detach().cpu()-expected[k]).abs().max())
                       for k, v in restored.state_dict().items() if v.dtype != torch.bool and v.numel())
    moment_error = 0.0
    for key, values in opt.state_dict()['state'].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                moment_error = max(moment_error, float((value - expected_optimizer['state'][key][name]).abs().max()))
    result = dict(loss_difference=abs(reference_loss-replay_loss),
                  model_max_abs_error=tensor_error, optimizer_max_abs_error=moment_error,
                  passed=tensor_error <= 1e-7 and moment_error <= 1e-7
                         and abs(reference_loss-replay_loss) <= 1e-7)
    del restored, opt, expected, expected_optimizer
    restore_state(model, optimizer, path, device)
    torch.cuda.empty_cache()
    if not result['passed']:
        raise RuntimeError(f'Resume replay failed: {result}')
    return result


def check_frozen(model, initial_state):
    changed = []
    count = 0
    for name, value in model.state_dict().items():
        if '.task_A.1' in name or '.task_B.1' in name:
            continue
        count += 1
        if not torch.equal(value.cpu(), initial_state[name]):
            changed.append(name)
    if changed:
        raise RuntimeError(f'Frozen weights/GPM changed: {changed[:10]}')
    return dict(tensors_checked=count, changed=changed, passed=True)


@torch.no_grad()
def check_merge(model, dataset, device):
    # TF32 may choose different rounded convolution paths for W*x + delta*x
    # versus (W+delta)*x. Test the algebra in strict FP32, without changing the
    # training/evaluation backend or weakening any numerical threshold.
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        result = _check_merge(model, dataset, device)
        result['precision'] = 'strict FP32 (cuDNN and matmul TF32 disabled for this check only)'
        return result
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


@torch.no_grad()
def _check_merge(model, dataset, device):
    pairs = []
    for index in selected_indices(len(dataset), 3):
        sample = dataset[index]
        x = sample['input'][:, :129, :131].unsqueeze(0).to(device)
        gt = sample['gt'][:, :129, :131].unsqueeze(0).to(device)
        pairs.append((x, gt, model(x, 1)))
    model.merge_task(1)
    rows = []
    for x, gt, before in pairs:
        after = model(x, -1)
        delta = (before-after).abs()
        rows.append(dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                         psnr_change=abs(compute_psnr(before, gt)-compute_psnr(after, gt))))
    passed = all(r['max_abs'] < 1e-3 and r['mean_abs'] < 1e-4 and r['psnr_change'] < 0.01 for r in rows)
    if not passed:
        raise RuntimeError(f'Merge check failed: {rows}')
    return dict(samples=rows, odd_input_size=[129, 131], passed=passed)


def datasets_from_manifest(manifest):
    records = manifest['records']
    def build(rows, augment=False):
        return RestorationDataset([r['input'] for r in rows], [r['gt'] for r in rows],
                                  patch_size=128, augment=augment)
    train = build(records['rainh_train'], augment=True)
    heldout = build(records['rainh_train'])
    tests = {task: build(records[f'{task}_test']) for task in ('lowlight', 'rainh')}
    return train, heldout, tests


def run_arm(args, manifest, output, arm):
    cfg = arm_config(manifest['config'], arm)
    device = torch.device('cuda:0')
    train_all, heldout, tests = datasets_from_manifest(manifest)
    train = Subset(train_all, manifest['train_indices'])
    full = {k: list(range(len(v))) for k, v in tests.items()}
    arm_dir = output / arm
    arm_dir.mkdir(exist_ok=True)
    (arm_dir / 'arm_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    criterion = TotalLoss(cfg).to(device).eval().requires_grad_(False)
    if not criterion.perceptual.use_lpips:
        raise RuntimeError('LPIPS required, no silent fallback')
    for seed in manifest['seeds']:
        run_dir = arm_dir / f'seed_{seed}'
        run_dir.mkdir(exist_ok=True)
        result_path = run_dir / 'result.json'
        if result_path.exists():
            print(f'SKIP completed {arm} seed={seed}', flush=True)
            continue
        started = time.monotonic()
        seed_all(seed)
        model, source = make_model(cfg, manifest['source_checkpoint'], device)
        if source.get('task_idx') != 0 or source.get('current_task_idx') != -1 or source.get('num_tasks') != 1:
            raise ValueError('Need a completed task-0 checkpoint, not merged.pth')
        model.add_task()
        model.set_current_task(1)
        seed_all(seed+101)
        ratios = []
        for i, batch in enumerate(DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)):
            ratios.append(freq_band_energy_from_bands(multilevel_dwt(batch['input'], 2)))
            if i == 4:
                break
        a, b = np.mean(ratios, axis=0), np.asarray(source['task_freq_energy']['lowlight'])
        similarity = float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        seed_all(seed+202)
        for i, batch in enumerate(DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)):
            model.calibrate(batch['input'].to(device), 1, similarity)
            if i == 5:
                break
        for name, tensor in source['model'].items():
            if not torch.equal(tensor, model.state_dict()[name]):
                raise RuntimeError(f'Calibration changed an old tensor: {name}')
        model.freeze_base_and_old(1)
        optimizer = optimizer_for(model, cfg)
        frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if '.task_A.1' not in k and '.task_B.1' not in k}
        del source
        latest = run_dir / 'latest.pth'
        best = run_dir / 'best_validation.pth'
        curve, best_psnr, best_step, start_step = [], -float('inf'), 0, 0
        if latest.exists():
            saved = restore_state(model, optimizer, latest, device)
            curve, best_psnr = saved['curve'], saved['best_psnr']
            best_step, start_step = saved['best_step'], saved['steps']
            del saved
        else:
            initial_path = arm_dir / 'initial_test_metrics.json'
            if not initial_path.exists():
                rows = evaluate(model, tests, full, device, criterion.perceptual.lpips, 0)
                write_json(initial_path, summarize_rows(rows))
        for start in range(start_step, args.steps, args.interval):
            # Each segment has an independent, predeclared seed. Resume needs
            # no hidden DataLoader cursor; augmentation is in the main process.
            seed_all(seed*100000 + start + 1000)
            loader = iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                     drop_last=True, num_workers=0))
            losses = []
            stop = min(start+args.interval, args.steps)
            for step in range(start+1, stop+1):
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                             drop_last=True, num_workers=0))
                    batch = next(loader)
                losses.append(train_step(model, optimizer, criterion, batch, cfg, device))
                if step % 50 == 0:
                    print(f'{arm} seed={seed} step={step}/{args.steps} loss={losses[-1]:.5f}', flush=True)
            validation_psnr = psnr_only(model, heldout, manifest['validation_indices'], device)
            old_diagnostic = psnr_only(model, tests['lowlight'], selected_indices(len(tests['lowlight']), 10), device)
            diag = update_diagnostics(model)
            if (any(r['gpm_rank'] == 0 for r in diag['layers'].values())
                    or diag['update_overlap_max'] > 1e-5):
                raise RuntimeError('Missing old GPM or excessive protected-update leakage')
            curve.append(dict(steps=stop, train_loss=float(np.mean(losses)),
                validation_psnr=validation_psnr, old_test_diagnostic_psnr=old_diagnostic,
                update_overlap_mean=diag['update_overlap_mean']))
            if validation_psnr > best_psnr:
                best_psnr, best_step = validation_psnr, stop
                save_state(best, model, optimizer, stop, curve, best_step, best_psnr)
            save_state(latest, model, optimizer, stop, curve, best_step, best_psnr)
            write_json(run_dir / 'curve.json', curve)
            write_json(arm_dir / 'status.json', dict(arm=arm, seed=seed, steps=stop,
                       elapsed_seconds=time.monotonic()-started, validation_psnr=validation_psnr))
            print(f'VALID {arm} seed={seed} step={stop} rainh_holdout={validation_psnr:.4f} '
                  f'old_diagnostic={old_diagnostic:.4f} overlap={diag["update_overlap_mean"]:.3g}', flush=True)
            if start == 0:
                check = check_resume(model, optimizer, criterion, cfg, latest, batch, device)
                write_json(run_dir / 'resume_replay.json', check)
        integrity = check_frozen(model, frozen)
        write_json(run_dir / 'final_diagnostics.json', update_diagnostics(model))
        final_rows = evaluate(model, tests, full, device, criterion.perceptual.lpips, 0)
        write_json(run_dir / 'final_test_rows.json', final_rows)
        final_metrics = summarize_rows(final_rows)
        final_merge = check_merge(model, tests['lowlight'], device)
        # Merging changes nonserialized runtime flags. Reconstruct instead of
        # reusing the merged object for checkpoint-selected evaluation.
        del model, optimizer
        torch.cuda.empty_cache()
        model, _ = make_model(cfg, str(best), device)
        model.set_current_task(1)
        if best_step == args.steps:
            selected_metrics = final_metrics
        else:
            selected_rows = evaluate(model, tests, full, device, criterion.perceptual.lpips, 0)
            write_json(run_dir / 'selected_test_rows.json', selected_rows)
            selected_metrics = summarize_rows(selected_rows)
        selected_merge = check_merge(model, tests['lowlight'], device)
        result = dict(seed=seed, arm=arm, final=final_metrics, selected=selected_metrics,
            selected_step=best_step, selected_validation_psnr=best_psnr,
            frozen_integrity=integrity, final_merge=final_merge, selected_merge=selected_merge,
            elapsed_seconds=time.monotonic()-started)
        write_json(result_path, result)
        del model, frozen
        torch.cuda.empty_cache()
        print('RUN COMPLETE', json.dumps(result), flush=True)
    write_json(arm_dir / 'status.json', dict(status='complete', arm=arm))


def report(output, manifest):
    results = {arm: [json.loads((output / arm / f'seed_{seed}' / 'result.json').read_text())
                     for seed in manifest['seeds']] for arm in ARMS}
    summary = dict(arms={}, paired_deltas={})
    lines = ['# 有效基与私有秩：分组验证、多种子和更长预算', '',
        f'RainH：{len(manifest["train_indices"])} 张用于校准/优化，'
        f'{len(manifest["validation_indices"])} 张（{manifest["validation_groups"]} 个不同 GT）用于选择 checkpoint。',
        '训练/验证按解码后 GT 内容分组；测试 GT 与训练 GT 无完全相同内容。'
        '这不保证近重复场景完全隔离。', '',
        f'增量种子 {manifest["seeds"]}；每组 {manifest["steps"]} 更新，batch={manifest["batch_size"]}；'
        '同一 task-0 起点、相同数据/校准种子、固定学习率与损失；GPM-32 不变。', '',
        '## 完整测试集结果（跨种子 mean ± sample std）', '',
        '| checkpoint | 方法 | LSRW PSNR | RainH PSNR | 两任务平均 PSNR | 平均 SSIM | 平均 LPIPS |',
        '|---|---|---:|---:|---:|---:|---:|']
    def stat(values):
        return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else 0.)
    def format_stat(s):
        return f'{s["mean"]:.4f} ± {s["std"]:.4f}'
    for arm in ARMS:
        summary['arms'][arm] = {}
        for kind in ('final', 'selected'):
            rows = results[arm]
            stats = {task: {metric: stat([r[kind][task][metric] for r in rows])
                           for metric in ('psnr', 'ssim', 'lpips')} for task in ('lowlight', 'rainh')}
            avg = {metric: stat([np.mean([r[kind][t][metric] for t in ('lowlight','rainh')])
                                for r in rows]) for metric in ('psnr','ssim','lpips')}
            summary['arms'][arm][kind] = dict(tasks=stats, average=avg)
            lines.append(f'| {kind} | {arm} | {format_stat(stats["lowlight"]["psnr"])} | '
                f'{format_stat(stats["rainh"]["psnr"])} | {format_stat(avg["psnr"])} | '
                f'{format_stat(avg["ssim"])} | {format_stat(avg["lpips"])} |')
    for kind in ('final', 'selected'):
        deltas = [float(np.mean([b[kind][t]['psnr']-a[kind][t]['psnr'] for t in ('lowlight','rainh')]))
                  for a, b in zip(results['current_basis'], results['effective_r16'])]
        summary['paired_deltas'][kind] = dict(per_seed=deltas, **stat(deltas),
                                             positive_seeds=sum(x>0 for x in deltas))
    lines += ['', '## 逐种子结果与模型选择', '',
              '| 方法 | seed | 验证最优步数 | 验证 PSNR | final LSRW | final RainH |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, rows in results.items():
        for row in rows:
            lines.append(f'| {arm} | {row["seed"]} | {row["selected_step"]} | '
                f'{row["selected_validation_psnr"]:.4f} | {row["final"]["lowlight"]["psnr"]:.4f} | '
                f'{row["final"]["rainh"]["psnr"]:.4f} |')
    for kind, row in summary['paired_deltas'].items():
        lines += ['', f'{kind}：候选减基线的两任务平均 PSNR = {format_stat(row)} dB；'
                  f'{row["positive_seeds"]}/{len(manifest["seeds"])} 个种子为正。']
    baseline = summary['arms']['current_basis']['selected']
    candidate = summary['arms']['effective_r16']['selected']
    original = json.loads((output / 'current_basis' / 'initial_test_metrics.json').read_text())
    old_difference = candidate['tasks']['lowlight']['psnr']['mean'] - baseline['tasks']['lowlight']['psnr']['mean']
    new_difference = candidate['tasks']['rainh']['psnr']['mean'] - baseline['tasks']['rainh']['psnr']['mean']
    old_forgetting = candidate['tasks']['lowlight']['psnr']['mean'] - original['lowlight']['psnr']
    lines += ['', '## 结论与下一步', '',
        f'以训练划出的分组验证集选模：候选较基线 RainH PSNR {new_difference:+.4f} dB，'
        f'LSRW {old_difference:+.4f} dB；候选相对原 task-0 的 LSRW 变化为 {old_forgetting:+.4f} dB。'
        '因此是整体指标更好的候选，而不是消除了遗忘。',
        f'两任务平均 SSIM {baseline["average"]["ssim"]["mean"]:.4f} → '
        f'{candidate["average"]["ssim"]["mean"]:.4f}；平均 LPIPS '
        f'{baseline["average"]["lpips"]["mean"]:.4f} → {candidate["average"]["lpips"]["mean"]:.4f}。',
        '优先在修复配对后、使用训练集独立验证划分，从头重训 LSRW task-0（新输出目录）。'
        '然后复验 RainH，包含等参数 effective-r8 与增容 r16，最后再正确更新任务末 GPM 并接 Haze4K。'
        '本轮没有启动这些 400 轮正式长跑，也没有把候选替换为正式默认。']
    replay = [json.loads((output / arm / f'seed_{s}' / 'resume_replay.json').read_text())
              for arm in ARMS for s in manifest['seeds']]
    summary['verification'] = dict(resume_replay_passed=all(r['passed'] for r in replay),
                                   resume_replays=len(replay))
    lines += ['', f'断点重放：{sum(r["passed"] for r in replay)}/{len(replay)} 运行通过；'
              f'模型最大绝对误差 {max(r["model_max_abs_error"] for r in replay):.3g}，'
              f'优化器矩最大误差 {max(r["optimizer_max_abs_error"] for r in replay):.3g}。']
    export_path = output / 'strict_fp32_export_checks.json'
    if export_path.exists():
        exports = json.loads(export_path.read_text())
        samples = [r for arm in exports['results'].values() for seed in arm.values()
                   for filename in ('latest.pth', 'best_validation.pth')
                   for r in seed[filename]['merge']['samples']]
        summary['verification']['strict_fp32_exports_passed'] = exports['passed']
        lines += ['', f'统一严格 FP32 导出复核：{exports["checkpoints"]} 份 checkpoint 全部通过；'
                  f'三张固定奇数裁剪的全局最大像素误差 {max(r["max_abs"] for r in samples):.3g}，'
                  f'最大 PSNR 差 {max(r["psnr_change"] for r in samples):.3g} dB。',
                  '默认 cuDNN TF32 路径曾在基线 seed42、候选 seed123 超出原合并误差阈值；'
                  '原始日志和 default_backend_merge_failure_*.json 均保留。'
                  '严格 FP32 仅用于代数复核，未放宽阈值、未改变主训练与完整测试精度，'
                  '也未重做任何已完成的训练更新。']
    coverage_path = output / 'coverage_seed7_summary.json'
    if coverage_path.exists():
        cover = json.loads(coverage_path.read_text())['results']
        lines += ['', f'首个种子的旧 GPM 抽样覆盖率：训练前 {cover["before"]["mean"]:.2%}，'
                  f'基线后 {cover["current_basis"]["mean"]:.2%}，候选后 {cover["effective_r16"]["mean"]:.2%}。'
                  '没有明显坍塌，但原先就没有覆盖全部旧特征。'
                  '这是 8 张中心裁剪、逐层非加权能量覆盖均值，不是遗忘百分比或全局精度。']
    lines += ['', '固定例图及视觉局限见 [VISUAL_REVIEW.md](VISUAL_REVIEW.md)；'
              '复现命令和精度说明见 configs/BASIS_VALIDATION.md。']
    lines += ['', '## 完整性与解释边界', '',
        '- 每个运行均进行真实模型/AdamW 状态保存恢复后的一步重放，检查 loss、参数和优化器矩一致。',
        '- 所有旧任务参数、基座、共享基和 GPM 必须逐位不变；完整更新的旧空间泄漏另存逐层诊断。',
        '- final 与验证最优模型均在 3 张固定 129×131 奇数裁剪上检查合并前后像素和 PSNR 误差。',
        '- 曲线中的旧任务 10 张测试图只用于保留诊断，不参与选模；新任务选模仅使用训练划出的分组验证集。',
        '- 已修复 LSRW 配对：5600 张曾因补零编号碰撞被笛卡尔展开为 10500 对；'
        '其中4900对是不同文件名的错误配对。旧 checkpoint 的训练过程无法凭当前权重追溯确认，'
        '因此本轮是固定旧起点的算法复现，不是数据修正后的从零基准。必须另行干净重训任务0。',
        '- 测试集此前已参与算法开发；本轮不是全新、完全未接触测试集的无偏确认。',
        '- 仅改变增量阶段种子，没有多种子重训 LSRW；3 种子不足以作强统计显著性结论。',
        '- effective-r16 比基线多参数；不是等参数对比。1000步不等于400轮，不代表三任务或五任务结果。',
        '- 本轮冻结任务末 GPM；latest/best_validation.pth 均为诊断 checkpoint，不直接作为下一任务起点。', '',
        '![分组验证和旧任务保留曲线](curves.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in ARMS:
        curves = [json.loads((output / arm / f'seed_{s}' / 'curve.json').read_text()) for s in manifest['seeds']]
        steps = [r['steps'] for r in curves[0]]
        for ax, key in zip(axes, ('validation_psnr', 'old_test_diagnostic_psnr')):
            values = np.array([[r[key] for r in curve] for curve in curves])
            mean, std = values.mean(0), values.std(0, ddof=1) if len(curves)>1 else np.zeros(len(steps))
            ax.plot(steps, mean, marker='o', label=arm)
            ax.fill_between(steps, mean-std, mean+std, alpha=.15)
            ax.set_xlabel('Optimizer steps')
            ax.set_ylabel('PSNR (dB), mean ± sample std')
            ax.grid(alpha=.3)
            ax.legend()
    axes[0].set_title('RainH grouped train holdout (selection)')
    axes[1].set_title('LSRW fixed 10 test images (diagnostic only)')
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=150)
    plt.close(fig)
    write_json(output / 'status.json', dict(status='complete', runs=len(ARMS)*len(manifest['seeds'])))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lsrw_rainh.yaml')
    parser.add_argument('--checkpoint', default='outputs/lsrw/task_0_final.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 42, 123])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--memory-fraction', type=float, default=.32)
    args = parser.parse_args()
    if min(args.steps, args.interval, args.batch_size) < 1 or args.steps % args.interval:
        parser.error('Positive budgets required; steps must be divisible by interval')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg['train']['lr'] != 2e-4 or cfg['train']['loss'] != {'l1': 1., 'perceptual': .1}:
        raise ValueError('Validation protocol requires fixed 2e-4, L1+0.1LPIPS')
    output = Path(args.output_dir)
    if args.prepare:
        prepare(args, cfg, output)
        return
    manifest = json.loads((output / 'manifest.json').read_text())
    for key in ('steps', 'interval', 'batch_size', 'seeds'):
        if getattr(args, key) != manifest[key]:
            raise ValueError(f'Protocol differs from prepared manifest: {key}')
    if cfg != manifest['config'] or file_sha(args.checkpoint) != manifest['source_sha256']:
        raise ValueError('Source checkpoint/config changed')
    for rows in manifest['records'].values():
        for row in rows:
            stats = [list((os.stat(p).st_size, os.stat(p).st_mtime_ns)) for p in (row['input'], row['gt'])]
            if stats != row['file_stats']:
                raise ValueError(f'Dataset changed after preparation: {row["input"]}')
    if args.report:
        report(output, manifest)
        return
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for arm in args.arms:
        run_arm(args, manifest, output, arm)


if __name__ == '__main__':
    main()
