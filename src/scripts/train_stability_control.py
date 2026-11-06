"""Short matched Haze4K control vs full-image replay/output-consistency experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import load_manifest, sha256_file
from src.trainers.clean_sequence_trainer import run_protocol, validate_resume
from src.trainers.stability_trainer import stability_config, StabilityTrainer
from src.utils.training_state import atomic_json, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/clean_effective_r16/task_1_final.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--mode', choices=('control','replay_distill'), default='replay_distill')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--memory-per-task', type=int, default=64)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.memory_per_task) < 1:
        parser.error('epochs / batch-size / memory-per-task 必须为正')
    cfg = stability_config(args.source, args.output_dir, args.mode, args.epochs,
                           args.batch_size, args.memory_per_task, args.smoke_test)
    if not torch.cuda.is_available():
        raise RuntimeError('此实验需要 CUDA 环境')
    manifest = load_manifest(cfg)
    split_sha = sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('stability_control_version') != 1:
            raise ValueError('--resume 只接受此入口的断点；共同起点请用 --source')
        validate_resume(cfg, saved, split_sha)
    output = Path(args.output_dir)
    run_path = output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not run_path.is_file():
            raise FileExistsError('请使用新输出目录，或指定本实验 --resume')
        existing = json.loads(run_path.read_text())
        if existing['protocol'] != run_protocol(cfg) or existing['split_sha256'] != split_sha:
            raise ValueError('目标目录属于不同实验')
    seed_all(cfg['experiment']['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(cfg['experiment']['memory_fraction'])
    atomic_json(run_path, dict(protocol=run_protocol(cfg), split_sha256=split_sha,
        resume=args.resume, source=cfg['stability']['source'],
        source_sha256=cfg['stability']['source_sha256'],
        source_code_sha256={p:sha256_file(p) for p in (
            'src/trainers/stability_trainer.py', 'src/trainers/clean_sequence_trainer.py',
            'src/scripts/train_stability_control.py')},
        note='Matched short-budget Haze4K diagnostic; full old TRAIN images; '
             'validation macro PSNR selection, never TEST; no architecture changes'))
    trainer = StabilityTrainer(cfg, manifest)
    trainer.run(stop_after='dehaze', resume_path=args.resume or args.source)
    if sha256_file(args.source) != cfg['stability']['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）', flush=True)


if __name__ == '__main__':
    main()
