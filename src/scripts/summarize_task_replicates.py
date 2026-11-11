"""Validate completed paired Haze4K seeds; summarize paired differences, not best seeds."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from src.data.clean_protocol import sha256_file
from src.models.task_memory import bank_hash
from src.trainers.memory_source_trainer import non_memory_state_fingerprint
from src.trainers.task_replicate_trainer import validate_rainh_source
from src.utils.training_state import atomic_json

TASKS = ('lowlight', 'rainh', 'dehaze')


def normalize_protocol(protocol):
    result = copy.deepcopy(protocol)
    for k in ('output_dir', 'name', 'incremental_seed'):
        result['experiment'].pop(k, None)
    for k in ('mode', 'selector'):
        result['task_memory'].pop(k, None)
    result.pop('task_replicate', None)
    return result


def vector(metrics):
    row = metrics['matrix']['2']
    result = {f'{t}_{m}': row[t][m] for t in TASKS for m in ('psnr', 'ssim', 'lpips')}
    result.update(mean_psnr=metrics['latest']['acc_psnr'], forgetting_psnr=metrics['latest']['forgetting_psnr'],
                  mean_ssim=float(np.mean([row[t]['ssim'] for t in TASKS])),
                  mean_lpips=float(np.mean([row[t]['lpips'] for t in TASKS])))
    return result


def summarize_runs(paths, source):
    source_state = torch.load(source, map_location='cpu', weights_only=False)
    validate_rainh_source(source_state)
    physical = non_memory_state_fingerprint(source_state['model'])
    bank = bank_hash(source_state['task_sketches'])
    expected = normalize_protocol(source_state['run_protocol'])
    by_seed = {}; provenance = {}; reference_code = None; replicate_code = None
    source_sha = sha256_file(source)
    provenance[str(source)] = source_sha
    for path in paths:
        path = Path(path)
        run = json.loads((path/'run_manifest.json').read_text())
        cfg = run['protocol']
        if cfg.get('smoke_test') or normalize_protocol(cfg) != expected:
            raise ValueError(f'Smoke or incompatible protocol: {path}')
        if run['split_sha256'] != source_state['split_sha256']:
            raise ValueError(f'Partition mismatch: {path}')
        code = run.get('source_code_sha256', {})
        core = ('src/models/task_memory.py', 'src/models/odr_lora.py', 'src/models/restoration_net.py',
                'src/trainers/task_balanced_trainer.py', 'src/trainers/memory_source_trainer.py',
                'src/trainers/clean_sequence_trainer.py')
        if not all(k in code for k in core):
            raise ValueError(f'Missing core code provenance: {path}')
        current_code = {k: code[k] for k in core}
        if reference_code is not None and current_code != reference_code:
            raise ValueError(f'Core training code changed between runs: {path}')
        reference_code = current_code
        status = json.loads((path/'status.json').read_text())
        if status.get('status') != 'complete' or status.get('stopped_after') != 'dehaze' or status.get('smoke_test'):
            raise ValueError(f'Incomplete or smoke run: {path}')
        metrics = json.loads((path/'continual_metrics.json').read_text())
        if metrics['latest']['stage'] != 2 or metrics['matrix']['1'] != source_state['performance_matrix']['1']:
            raise ValueError(f'Not the common RainH evaluation history: {path}')
        reported = metrics['latest']['acc_psnr']
        actual = float(np.mean([metrics['matrix']['2'][t]['psnr'] for t in TASKS]))
        if not np.isfinite(actual) or abs(actual-reported)>1e-7:
            raise ValueError(f'Inconsistent final macro PSNR: {path}')
        boundary = torch.load(path/'task_1_final.pth', map_location='cpu', weights_only=False)
        if (boundary.get('in_progress') or boundary.get('task_idx') != 1 or boundary.get('smoke_test')
                or not boundary.get('gpm_updated') or boundary.get('task_balanced_version') != 1
                or non_memory_state_fingerprint(boundary['model']) != physical
                or bank_hash(boundary['task_sketches']) != bank
                or boundary.get('task_sketches_sha256') != bank):
            raise ValueError(f'Not the common physical RainH model and snapshots: {path}')
        if boundary['run_protocol'] != cfg:
            raise ValueError(f'Checkpoint/manifest protocol mismatch: {path}')
        replicate = cfg.get('task_replicate')
        if replicate:
            if replicate_code is not None and code != replicate_code:
                raise ValueError(f'Replicate implementation changed between new runs: {path}')
            replicate_code = code
        seed = cfg['experiment']['incremental_seed']
        if replicate and (replicate['seed'] != seed or replicate['effective_haze_seed'] != seed+20000
                          or replicate['source_sha256'] != source_sha
                          or replicate['source_physical_sha256'] != physical
                          or replicate['source_bank_sha256'] != bank):
            raise ValueError(f'Replicate source or seed mismatch: {path}')
        mode = cfg['task_memory']['mode']
        if mode not in ('balanced', 'mean_pca') or mode in by_seed.setdefault(seed, {}):
            raise ValueError(f'Duplicate seed/mode or unknown mode: {path}')
        by_seed[seed][mode] = dict(path=str(path), metrics=vector(metrics))
        for name in ('run_manifest.json', 'continual_metrics.json', 'task_1_final.pth', 'status.json'):
            provenance[str(path/name)] = sha256_file(path/name)
    if not by_seed or any(set(pair) != {'balanced', 'mean_pca'} for pair in by_seed.values()):
        raise ValueError('Every seed must have exactly one completed balanced/mean_pca pair')
    pairs = []
    for seed, pair in sorted(by_seed.items()):
        delta = {k: pair['balanced']['metrics'][k]-pair['mean_pca']['metrics'][k]
                 for k in pair['balanced']['metrics']}
        pairs.append(dict(seed=seed, effective_haze_seed=seed+20000, **pair, delta=delta))
    stats = {}
    for key in pairs[0]['delta']:
        values = [p['delta'][key] for p in pairs]
        stats[key] = dict(mean=float(np.mean(values)),
                          sample_std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                          positive_seeds=sum(x>0 for x in values), values=values)
    return dict(n=len(pairs), pairs=pairs, paired_difference_stats=stats, sources_sha256=provenance,
        shared_core_code_sha256=reference_code,
        definition='delta = balanced - mean_pca; higher PSNR/SSIM and lower LPIPS/forgetting are better',
        scope='Conditional Haze4K seed replication given fixed LSRW/RainH weights and frozen snapshots',
        caution='Observed seed42 may be included as exploratory reference. Small n is not proof of statistical significance.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/task_balanced_b64/task_1_final.pth')
    p.add_argument('--root', default='outputs/task_memory_replicates')
    p.add_argument('--seeds', type=int, nargs='+', default=[7, 2026])
    p.add_argument('--include-existing', action='store_true', help='Also include the original seed42 formal pair')
    p.add_argument('--output-dir', default='outputs/task_memory_replicates/summary')
    a = p.parse_args()
    torch.set_num_threads(4)
    paths = [Path(a.root)/f'{mode}_seed{seed}' for seed in a.seeds for mode in ('balanced', 'mean_pca')]
    if a.include_existing:
        paths += [Path('outputs/task_balanced_b64'), Path('outputs/task_mean_pca_b64')]
    result = summarize_runs(paths, a.source)
    output = Path(a.output_dir)
    atomic_json(output/'summary.json', result)
    keys = list(result['pairs'][0]['delta'])
    with (output/'paired_deltas.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'effective_haze_seed', *keys])
        writer.writeheader()
        for pair in result['pairs']:
            writer.writerow(dict(seed=pair['seed'], effective_haze_seed=pair['effective_haze_seed'], **pair['delta']))
    with (output/'all_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'mode', 'path', *keys])
        writer.writeheader()
        for pair in result['pairs']:
            for mode in ('balanced', 'mean_pca'):
                writer.writerow(dict(seed=pair['seed'], mode=mode, path=pair[mode]['path'], **pair[mode]['metrics']))
    lines = ['# 固定RainH边界的配对种子复验', '', f"完整配对数：{result['n']}。差值为 balanced − mean_pca。", '',
             '| 种子 | 平衡组平均PSNR | 对照平均PSNR | 差值/dB |', '|---|---:|---:|---:|']
    for pair in result['pairs']:
        lines.append(f"| {pair['seed']} | {pair['balanced']['metrics']['mean_psnr']:.4f} | "
                     f"{pair['mean_pca']['metrics']['mean_psnr']:.4f} | {pair['delta']['mean_psnr']:+.4f} |")
    lines += ['', '| 配对差值指标 | 均值 | 样本标准差 |', '|---|---:|---:|']
    for key in ('lowlight_psnr', 'rainh_psnr', 'dehaze_psnr', 'mean_psnr', 'forgetting_psnr'):
        stat = result['paired_difference_stats'][key]
        deviation = '—' if stat['sample_std'] is None else f"{stat['sample_std']:.4f}"
        lines.append(f"| {key} | {stat['mean']:+.4f} | {deviation} |")
    lines += ['', '只复验固定前两任务后的Haze4K随机性，不代表整条序列的多种子结果。',
              'PSNR差值越高越好，遗忘差值越低越好；样本标准差不是置信区间，不据小样本宣称统计显著。',
              '如含seed42，它是此前已观察的探索性结果；应同时查看两个新增种子的表现。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    metric = result['paired_difference_stats']['mean_psnr']
    print(f"n={result['n']}; paired mean PSNR delta={metric['mean']:+.6f} dB; sample SD={metric['sample_std']}")
    print(f'Validated summary: {output.resolve()}')


if __name__ == '__main__': main()
