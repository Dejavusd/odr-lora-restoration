"""Clean LSRW -> RainH -> Haze4K training, with safe stage and epoch resume."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import yaml

from src.data.clean_protocol import load_manifest, sha256_file
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, run_protocol, validate_resume
from src.utils.training_state import atomic_json, seed_all


def configure_method(cfg, method):
    cfg = copy.deepcopy(cfg)
    cfg['experiment']['method'] = method
    cfg['model']['new_task_private_rank_base'] = 16 if method == 'effective_r16' else 8
    cfg['odr_lora'].update(full_update_protection=True,
        basis_construction='legacy' if method == 'current_basis' else 'projected_svd')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_lsrw_rainh_haze4k.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume')
    parser.add_argument('--allow-batch-size-change', action='store_true',
                        help='Allow batch size / memory quota changes at a completed task; use a fresh output directory')
    parser.add_argument('--stop-after', choices=('lowlight', 'rainh', 'dehaze'))
    parser.add_argument('--method', choices=('current_basis', 'effective_r8', 'effective_r16'), default='effective_r16')
    parser.add_argument('--incremental-seed', type=int)
    parser.add_argument('--smoke-test', action='store_true', help='Isolated 2-step/task wiring test; NOT a formal run')
    args = parser.parse_args()
    if args.allow_batch_size_change and not args.resume:
        parser.error('--allow-batch-size-change 需要 --resume 指向已完成任务的断点')
    cfg = configure_method(yaml.safe_load(Path(args.config).read_text()), args.method)
    if args.output_dir:
        cfg['experiment']['output_dir'] = args.output_dir
    if args.incremental_seed is not None:
        cfg['experiment']['incremental_seed'] = args.incremental_seed
    cfg['experiment'].setdefault('incremental_seed', cfg['experiment']['seed'])
    cfg['smoke_test'] = args.smoke_test
    if args.smoke_test:
        if not args.output_dir:
            parser.error('--smoke-test 必须显式设置独立 --output-dir')
        cfg['train'].update(epochs_per_task=1, max_steps_per_epoch=2, calibration_batches=1, gpm_batches=1)
        cfg['eval'].update(validation_interval=1, max_images=2)
        cfg['data']['num_workers'] = 0
        for task in cfg['data']['tasks']:
            task['batch_size'] = 2
    if [t['name'] for t in cfg['data']['tasks']] != ['lowlight', 'rainh', 'dehaze']:
        raise ValueError('该入口固定用于 lowlight -> rainh -> dehaze')
    if cfg['train'].get('scheduler') != 'cosine' or cfg['train'].get('warmup_epochs', 0) != 0:
        raise ValueError('本协议使用无 warmup 的 cosine scheduler，不静默忽略其他设置')
    if min(cfg['train'][k] for k in ('epochs_per_task', 'calibration_batches', 'gpm_batches')) < 1:
        raise ValueError('训练与校准预算必须为正')
    if cfg['eval']['validation_interval'] < 1:
        raise ValueError('validation_interval 必须为正')
    if any(t['batch_size'] < 1 or t['patch_size'] < 16 for t in cfg['data']['tasks']):
        raise ValueError('batch_size 必须为正；patch_size 至少16')
    device = cfg['experiment'].get('device', 'cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA 训练但当前不可用；请先检查 odr-lora 环境')
    memory_fraction = cfg['experiment'].get('memory_fraction', .5)
    if not 0 < memory_fraction <= 1:
        raise ValueError('memory_fraction 必须在 (0,1]')
    manifest = load_manifest(cfg)
    split_sha = sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        # Preflight before creating output directories or GPU models.
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume(cfg, state, split_sha, args.allow_batch_size_change)
        if args.stop_after and state['task_idx'] > ['lowlight','rainh','dehaze'].index(args.stop_after):
            raise ValueError('checkpoint 已超过 stop-after 指定的阶段，不能倒退')
        del state
    output = Path(cfg['experiment']['output_dir'])
    run_path = output / 'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if args.allow_batch_size_change:
            raise FileExistsError('更改 batch size 分支必须使用新的空输出目录')
        if not args.resume or not run_path.exists():
            raise FileExistsError(f'输出目录非空，拒绝覆盖：{output}；续训请指定 --resume')
        existing = json.loads(run_path.read_text())
        if existing['protocol'] != run_protocol(cfg) or existing['split_sha256'] != split_sha:
            raise ValueError('目标输出目录属于不同实验，请使用新的 --output-dir')
    seed_all(cfg['experiment']['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg['experiment'].get('strict_fp32', True):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith('cuda'):
        index = torch.device(device).index
        torch.cuda.set_per_process_memory_fraction(memory_fraction,
            device=torch.cuda.current_device() if index is None else index)
    atomic_json(run_path, dict(protocol=run_protocol(cfg), split_sha256=split_sha,
        config_source=str(Path(args.config).resolve()), resume=args.resume,
        resume_sha256=sha256_file(args.resume) if args.resume else None,
        allow_batch_size_change=args.allow_batch_size_change,
        smoke_test=args.smoke_test, source_sha256={p:sha256_file(p) for p in (
            'src/data/datasets.py', 'src/data/clean_protocol.py',
            'src/trainers/clean_sequence_trainer.py', 'src/scripts/train_clean_sequence.py')},
        metric_protocol='Full-image RGB PSNR/SSIM clamped; existing raw-pred LPIPS VGG; '
                        'checkpoint selection only on grouped TRAIN holdout; tests at task boundaries',
        precision='strict FP32' if cfg['experiment'].get('strict_fp32', True) else 'default backend'))
    trainer = CleanSequenceTrainer(cfg, manifest, args.allow_batch_size_change)
    trainer.run(stop_after=args.stop_after, resume_path=args.resume)
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）', flush=True)


if __name__ == '__main__':
    main()
