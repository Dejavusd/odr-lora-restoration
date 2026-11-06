"""Replay-free Haze4K experiment: hard orthogonality plus bounded LoRA updates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import load_manifest, sha256_file
from src.trainers.bounded_lora_trainer import bounded_config, BoundedLoRATrainer
from src.trainers.clean_sequence_trainer import run_protocol, validate_resume
from src.utils.training_state import atomic_json, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/clean_effective_r16/task_1_final.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-relative-update', type=float, default=.05,
                        help='Per-layer ||A B||_F / max(||W||_F,1e-12); 0 disables cap')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cfg = bounded_config(args.source,args.output_dir,args.max_relative_update,
                         args.epochs,args.batch_size,args.smoke_test)
    if not torch.cuda.is_available():
        raise RuntimeError('训练需要 CUDA')
    manifest = load_manifest(cfg)
    split_sha = sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('bounded_lora_version') != 1:
            raise ValueError('--resume 只接受本入口断点，原始共同起点请用 --source')
        validate_resume(cfg,saved,split_sha)
    output = Path(args.output_dir)
    run_path = output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not run_path.is_file():
            raise FileExistsError('请使用新输出目录，或指定本实验 --resume')
        saved = json.loads(run_path.read_text())
        if saved['protocol'] != run_protocol(cfg) or saved['split_sha256'] != split_sha:
            raise ValueError('目标目录属于不同实验')
    seed_all(cfg['experiment']['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(cfg['experiment']['memory_fraction'])
    atomic_json(run_path,dict(protocol=run_protocol(cfg),split_sha256=split_sha,resume=args.resume,
        source_sha256=cfg['update_bound']['source_sha256'],
        source_code_sha256={p:sha256_file(p) for p in (
            'src/trainers/bounded_lora_trainer.py','src/trainers/clean_sequence_trainer.py',
            'src/scripts/train_bounded_lora.py')},
        note='No replay/teacher/distillation/range loss; current-task training only. '
             'Old validation forward-only diagnostics; selection uses ONLY Haze4K validation PSNR.'))
    trainer = BoundedLoRATrainer(cfg,manifest)
    trainer.run(stop_after='dehaze',resume_path=args.resume or args.source)
    if sha256_file(args.source) != cfg['update_bound']['source_sha256']:
        raise RuntimeError('源断点变化')
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）',flush=True)


if __name__ == '__main__':
    main()
