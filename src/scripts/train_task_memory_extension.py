"""Continue stable per-task memory: DF2K denoising, optionally GOPRO deblurring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from src.data.clean_protocol import sha256_file
from src.data.five_task_protocol import load_extension
from src.trainers.clean_sequence_trainer import run_protocol, validate_resume
from src.trainers.task_memory_extension_trainer import task_memory_extension_config, TaskMemoryExtensionTrainer
from src.utils.training_state import atomic_json, seed_all


CODE_FILES = (
    'src/scripts/train_task_memory_extension.py', 'src/trainers/task_memory_extension_trainer.py',
    'src/trainers/task_balanced_trainer.py', 'src/trainers/five_task_trainer.py',
    'src/trainers/clean_sequence_trainer.py', 'src/trainers/memory_source_trainer.py',
    'src/trainers/incremental_trainer.py', 'src/data/five_task_protocol.py', 'src/data/clean_protocol.py',
    'src/data/transforms.py', 'src/data/datasets.py', 'src/models/task_memory.py',
    'src/models/backbone.py', 'src/models/odr_lora.py', 'src/models/restoration_net.py',
    'src/models/dual_gpm.py', 'src/utils/checkpoint.py', 'src/utils/training_state.py',
    'src/utils/inference.py', 'src/utils/metrics.py', 'src/losses/losses.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='This arm formal task_2_final.pth, not merged.pth')
    p.add_argument('--manifest', default='outputs/five_task_splits_v1/manifest.json')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume')
    p.add_argument('--stop-after', choices=('denoise', 'deblur'), default='denoise')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    cfg = task_memory_extension_config(a.source, a.manifest, a.output_dir,
                                       a.epochs, a.batch_size, a.smoke_test)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    manifest = load_extension(cfg)
    split_sha = sha256_file(a.manifest)
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        if state.get('task_memory_extension_version') != 1:
            raise ValueError('--resume accepts only this extension entry checkpoints; use --source for Haze4K')
        validate_resume(cfg, state, split_sha)
        if state['task_idx'] > [t['name'] for t in cfg['data']['tasks']].index(a.stop_after):
            raise ValueError('Cannot move backwards to an earlier task')
    output = Path(a.output_dir)
    run = output/'run_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not a.resume or not run.is_file():
            raise FileExistsError('Use a fresh output directory or --resume; never overwrite existing results')
        previous = json.loads(run.read_text())
        if previous['protocol'] != run_protocol(cfg) or previous['split_sha256'] != split_sha:
            raise ValueError('Output directory protocol mismatch')
        if any(sha256_file(ROOT/f) != digest for f, digest in previous['source_code_sha256'].items()):
            raise ValueError('Training implementation changed; cannot silently resume')
    seed_all(cfg['experiment']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    atomic_json(run, dict(protocol=run_protocol(cfg), split_sha256=split_sha, resume=a.resume,
        source_code_sha256={f: sha256_file(ROOT/f) for f in CODE_FILES},
        note='Own-arm Haze4K boundary; unchanged historical task summaries and GPM; no image replay. '
             'Default stops after denoise. New-task holdout only selects weights.'))
    trainer = TaskMemoryExtensionTrainer(cfg, manifest)
    trainer.run(stop_after=a.stop_after, resume_path=a.resume or a.source)
    if sha256_file(a.source) != cfg['task_memory_extension']['source_sha256']:
        raise RuntimeError('Haze4K source changed')
    print(f'完成：{output.resolve()}（smoke_test={a.smoke_test}）', flush=True)


if __name__ == '__main__':
    main()
