"""B ablation: normalized skip-branch memory versus fused-only, no replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import load_manifest, sha256_file
from src.trainers.clean_sequence_trainer import run_protocol, validate_resume
from src.trainers.skip_branch_trainer import skip_branch_config, SkipBranchTrainer
from src.utils.training_state import atomic_json, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/stable_v1_b64/task_0_final.pth')
    p.add_argument('--mode', choices=('fused', 'branch_v1'), default='branch_v1')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after', choices=('rainh', 'dehaze'), default='dehaze')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    cfg = skip_branch_config(args.source, args.output_dir, args.mode, args.epochs,
                             args.batch_size, args.smoke_test)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    manifest = load_manifest(cfg)
    split_sha = sha256_file(cfg['data']['split_manifest'])
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('skip_branch_version') != 1:
            raise ValueError('--resume only accepts this experiment checkpoints')
        validate_resume(cfg, saved, split_sha)
        if saved['task_idx'] > ['lowlight', 'rainh', 'dehaze'].index(args.stop_after):
            raise ValueError('Checkpoint is beyond stop-after')
    output = Path(args.output_dir)
    run = output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not run.is_file():
            raise FileExistsError('Use a fresh output directory or --resume')
        previous = json.loads(run.read_text())
        if previous['protocol'] != run_protocol(cfg) or previous['split_sha256'] != split_sha:
            raise ValueError('Output directory protocol mismatch')
    seed_all(cfg['experiment']['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    code = ('src/models/backbone.py', 'src/models/odr_lora.py', 'src/models/restoration_net.py',
            'src/trainers/skip_branch_trainer.py', 'src/trainers/memory_source_trainer.py',
            'src/trainers/clean_sequence_trainer.py', 'src/scripts/train_skip_branch.py')
    atomic_json(run, dict(protocol=run_protocol(cfg), split_sha256=split_sha, resume=args.resume,
                         source_code_sha256={f: sha256_file(f) for f in code},
                         note='Same LSRW weights, rebuild memory at task0 boundary; '
                              'then current-task TRAIN only; no replay, teacher, rank or architecture change.'))
    trainer = SkipBranchTrainer(cfg, manifest)
    trainer.run(stop_after=args.stop_after, resume_path=args.resume or args.source)
    if sha256_file(args.source) != cfg['memory_source']['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    print(f'完成：{output.resolve()}（smoke_test={args.smoke_test}）', flush=True)


if __name__ == '__main__':
    main()
