"""Uniform strict-FP32 merge audit of final and selected validation exports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.scripts.run_forgetting_probe import write_json
from src.scripts.validate_effective_basis import (
    ARMS, arm_config, check_merge, datasets_from_manifest, file_sha, make_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    manifest = json.loads((output / 'manifest.json').read_text())
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    _, _, datasets = datasets_from_manifest(manifest)
    results = {}
    for arm in ARMS:
        results[arm] = {}
        cfg = arm_config(manifest['config'], arm)
        for seed in manifest['seeds']:
            directory = output / arm / f'seed_{seed}'
            if not (directory / 'result.json').exists():
                raise RuntimeError(f'Run not complete: {directory}')
            checks, basis = {}, None
            for filename in ('latest.pth', 'best_validation.pth'):
                path = directory / filename
                model, state = make_model(cfg, str(path), device)
                current_basis = {k: v.cpu().clone() for k, v in state['model'].items() if '.task_B.1' in k}
                del state
                if basis is not None and any(not torch.equal(v, basis[k]) for k, v in current_basis.items()):
                    raise RuntimeError('Frozen B differs between final and selected checkpoints')
                basis = current_basis
                checks[filename] = dict(checkpoint_sha256=file_sha(path),
                    merge=check_merge(model, datasets['lowlight'], device))
                del model
                torch.cuda.empty_cache()
            checks['frozen_B_final_vs_selected_equal'] = True
            results[arm][str(seed)] = checks
            print(f'PASS strict-FP32 exports {arm} seed={seed}', flush=True)
    write_json(output / 'strict_fp32_export_checks.json', dict(
        passed=True, checkpoints=len(ARMS)*len(manifest['seeds'])*2,
        precision='Strict FP32 check only; original training/test backend unchanged',
        thresholds=dict(max_abs=1e-3, mean_abs=1e-4, psnr_change=.01), results=results))


if __name__ == '__main__':
    main()
