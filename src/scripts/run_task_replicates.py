"""Sequential paired Haze4K replications; restart the command to resume jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.clean_protocol import sha256_file
from src.trainers.clean_sequence_trainer import run_protocol
from src.trainers.task_replicate_trainer import task_replicate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/task_balanced_b64/task_1_final.pth')
    p.add_argument('--seeds', type=int, nargs='+', default=[7, 2026])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output-root', default='outputs/task_memory_replicates')
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    if len(a.seeds) != len(set(a.seeds)):
        p.error('Seeds must be unique')
    for seed in a.seeds:
        for mode in ('balanced', 'mean_pca'):
            out = Path(a.output_root)/f'{mode}_seed{seed}'
            cfg = task_replicate_config(a.source, out, mode, seed, a.epochs, a.batch_size, a.smoke_test)
            split_sha = sha256_file(cfg['data']['split_manifest'])
            run = out/'run_manifest.json'; latest = out/'latest.pth'; status = out/'status.json'
            resume = False
            if out.exists() and any(out.iterdir()):
                if not run.is_file():
                    raise FileExistsError(f'Nonempty unrelated directory: {out}')
                previous = json.loads(run.read_text())
                if previous['protocol'] != run_protocol(cfg) or previous['split_sha256'] != split_sha:
                    raise ValueError(f'Protocol mismatch: {out}')
                if status.is_file():
                    state = json.loads(status.read_text())
                    if (state.get('status') == 'complete' and state.get('stopped_after') == 'dehaze'
                            and state.get('smoke_test') == a.smoke_test
                            and (out/'task_2_final.pth').is_file() and (out/'continual_metrics.json').is_file()):
                        print(f'SKIP completed: {out}', flush=True)
                        continue
                if not latest.is_file():
                    raise FileExistsError(f'No complete checkpoint in {out}; preserve it and use a new output-root')
                resume = True
            command = [sys.executable, str(ROOT/'src/scripts/train_task_replicate.py'),
                '--source', a.source, '--mode', mode, '--seed', str(seed), '--epochs', str(a.epochs),
                '--batch-size', str(a.batch_size), '--output-dir', str(out)]
            if a.smoke_test: command.append('--smoke-test')
            if resume: command.extend(['--resume', str(latest)])
            print(f'{"RESUME" if resume else "START"}: {mode} seed={seed}, {out}', flush=True)
            subprocess.run(command, check=True)
    print('All requested paired runs completed. Use summarize_task_replicates.py to validate and aggregate.', flush=True)


if __name__ == '__main__': main()
