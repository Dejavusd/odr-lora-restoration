"""One paired-seed Haze4K run from a fixed formal RainH task-memory checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import load_manifest, sha256_file
from src.trainers.clean_sequence_trainer import run_protocol, validate_resume
from src.trainers.task_replicate_trainer import task_replicate_config, TaskReplicateTrainer
from src.utils.training_state import atomic_json, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/task_balanced_b64/task_1_final.pth')
    p.add_argument('--mode', choices=('balanced', 'mean_pca'), default='balanced')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume')
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    cfg = task_replicate_config(a.source, a.output_dir, a.mode, a.seed, a.epochs, a.batch_size, a.smoke_test)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    manifest = load_manifest(cfg); split_sha = sha256_file(cfg['data']['split_manifest'])
    if split_sha != cfg['task_replicate']['source_split_sha256']:
        raise ValueError('Source data partition changed')
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        if saved.get('task_replicate_version') != 1:
            raise ValueError('--resume requires this replicate entry checkpoints')
        validate_resume(cfg, saved, split_sha)
    output = Path(a.output_dir); run = output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not a.resume or not run.is_file():
            raise FileExistsError('Use a fresh output directory or --resume')
        previous = json.loads(run.read_text())
        if previous['protocol'] != run_protocol(cfg) or previous['split_sha256'] != split_sha:
            raise ValueError('Output directory protocol mismatch')
        if any(sha256_file(f) != digest for f, digest in previous['source_code_sha256'].items()):
            raise ValueError('Source code changed since this run; do not silently resume a different implementation')
    seed_all(cfg['experiment']['seed']); torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    code = ('src/trainers/task_replicate_trainer.py', 'src/trainers/task_balanced_trainer.py',
            'src/trainers/clean_sequence_trainer.py', 'src/trainers/memory_source_trainer.py', 'src/models/task_memory.py',
            'src/models/backbone.py', 'src/models/odr_lora.py', 'src/models/restoration_net.py',
            'src/scripts/train_task_replicate.py')
    atomic_json(run, dict(protocol=run_protocol(cfg), split_sha256=split_sha, resume=a.resume,
        source_code_sha256={f: sha256_file(f) for f in code},
        note='Fixed physical RainH weights and task snapshots; Haze4K seed replication only. '
             'Reselect shared GPM by mode without any old image access.'))
    trainer = TaskReplicateTrainer(cfg, manifest)
    trainer.run(stop_after='dehaze', resume_path=a.resume or a.source)
    if sha256_file(a.source) != cfg['task_replicate']['source_sha256']:
        raise RuntimeError('Common RainH source changed')
    print(f'完成：{output.resolve()}（smoke_test={a.smoke_test}）', flush=True)


if __name__ == '__main__': main()
