"""Compare completed own-arm denoising extensions, including incremental old-task drops."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.data.clean_protocol import sha256_file
from src.models.task_memory import bank_hash
from src.trainers.task_memory_extension_trainer import OLD_TASKS, validate_task_memory_source
from src.utils.training_state import atomic_json


TASKS = (*OLD_TASKS, 'denoise')


def comparable_protocol(protocol):
    result = copy.deepcopy(protocol)
    for key in ('name', 'output_dir'):
        result['experiment'].pop(key, None)
    for key in ('mode', 'selector'):
        result['task_memory'].pop(key, None)
    for key in ('mode', 'source', 'source_sha256', 'source_physical_sha256', 'source_bank_sha256'):
        result['task_memory_extension'].pop(key, None)
    return result


def summarize_extensions(paths):
    runs = {}
    reference = None
    code = None
    split_sha = None
    for path in map(Path, paths):
        manifest = json.loads((path/'run_manifest.json').read_text())
        cfg = manifest['protocol']
        status = json.loads((path/'status.json').read_text())
        if (cfg.get('smoke_test') or status.get('smoke_test') or status.get('status') != 'complete'
                or status.get('stopped_after') != 'denoise'):
            raise ValueError(f'Not a completed formal denoise extension: {path}')
        if not manifest.get('source_code_sha256'):
            raise ValueError('Missing implementation provenance')
        normalized = comparable_protocol(cfg)
        if reference is not None and (normalized != reference or code != manifest['source_code_sha256']
                                      or split_sha != manifest['split_sha256']):
            raise ValueError('Unmatched seed, budget, partition or implementation')
        reference, code, split_sha = normalized, manifest['source_code_sha256'], manifest['split_sha256']
        mode = cfg['task_memory']['mode']
        if mode not in ('balanced', 'mean_pca') or mode in runs:
            raise ValueError('Exactly one run per mode required')
        settings = cfg['task_memory_extension']
        source = Path(settings['source'])
        if sha256_file(source) != settings['source_sha256']:
            raise ValueError('Source SHA256 changed')
        source_state = torch.load(source, map_location='cpu', weights_only=False)
        validate_task_memory_source(source_state)
        if source_state['run_protocol']['task_memory']['mode'] != mode:
            raise ValueError('Source/extension memory mode mismatch')
        metrics = json.loads((path/'continual_metrics.json').read_text())
        matrix = metrics['matrix']
        if any(matrix[str(i)] != source_state['performance_matrix'][str(i)] for i in range(3)):
            raise ValueError('Historical evaluation records changed')
        if metrics['latest']['stage'] != 3 or set(matrix) != {'0', '1', '2', '3'}:
            raise ValueError('Expected four-task result')
        row, before = matrix['3'], matrix['2']
        values = {f'{task}_{metric}': row[task][metric]
                  for task in TASKS for metric in ('psnr', 'ssim', 'lpips')}
        values.update(mean_psnr=sum(row[t]['psnr'] for t in TASKS)/4,
                      old_mean_psnr=sum(row[t]['psnr'] for t in OLD_TASKS)/3,
                      old_mean_drop=sum(before[t]['psnr']-row[t]['psnr'] for t in OLD_TASKS)/3,
                      forgetting_psnr=metrics['latest']['forgetting_psnr'])
        for task in OLD_TASKS:
            values[f'{task}_drop'] = before[task]['psnr']-row[task]['psnr']
        groups = row['denoise']['datasets']
        if set(groups) != {'DIV2K', 'BSD68'}:
            raise ValueError('Missing denoising test subset')
        for name in ('DIV2K', 'BSD68'):
            values[f'denoise_{name}_psnr'] = groups[name]['psnr']
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite metrics')
        if abs(values['mean_psnr']-metrics['latest']['acc_psnr']) > 1e-7:
            raise ValueError('Inconsistent macro mean PSNR')
        final = torch.load(path/'task_3_final.pth', map_location='cpu', weights_only=False)
        if (final.get('task_memory_extension_version') != 1 or final.get('in_progress')
                or final.get('task_idx') != 3 or not final.get('gpm_updated') or final.get('smoke_test')
                or final['performance_matrix'] != matrix or final['run_protocol'] != cfg
                or final['split_sha256'] != split_sha
                or bank_hash(final['task_sketches']) != final['task_sketches_sha256']
                or set(final['task_sketches']) != set(source_state['task_sketches'])
                or any(set(entries) != {'0', '1', '2', '3'} for entries in final['task_sketches'].values())):
            raise ValueError('Invalid final task-memory checkpoint')
        runs[mode] = dict(path=str(path), metrics=values,
            provenance={str(path/name): sha256_file(path/name)
                        for name in ('run_manifest.json', 'continual_metrics.json', 'status.json')},
            source_sha256=settings['source_sha256'])
    if set(runs) != {'balanced', 'mean_pca'}:
        raise ValueError('Both balanced and mean_pca are required')
    return dict(runs=runs, delta={k: runs['balanced']['metrics'][k]-runs['mean_pca']['metrics'][k]
                                 for k in runs['balanced']['metrics']},
        note='balanced - mean_pca; lower LPIPS/forgetting/drop is better. Own-arm sequence continuation; '
             'different starting weights, not an isolated fourth-stage selector ablation. Single paired seed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--balanced', default='outputs/task_memory_denoise_b64/balanced')
    p.add_argument('--control', default='outputs/task_memory_denoise_b64/mean_pca')
    p.add_argument('--output-dir', default='outputs/task_memory_denoise_b64/summary')
    a = p.parse_args()
    torch.set_num_threads(4)
    report = summarize_extensions([a.balanced, a.control])
    output = Path(a.output_dir)
    atomic_json(output/'summary.json', report)
    lines = ['# 第四任务去噪：逐任务记忆匹配对照', '',
             '| 指标 | 平衡组 | PCA对照 | 差值 |', '|---|---:|---:|---:|']
    for key in ('lowlight_psnr', 'rainh_psnr', 'dehaze_psnr', 'denoise_psnr', 'denoise_DIV2K_psnr',
                'denoise_BSD68_psnr', 'mean_psnr', 'old_mean_psnr', 'old_mean_drop', 'forgetting_psnr'):
        lines.append(f"| {key} | {report['runs']['balanced']['metrics'][key]:.6f} | "
                     f"{report['runs']['mean_pca']['metrics'][key]:.6f} | {report['delta'][key]:+.6f} |")
    lines += ['', 'drop是各组相对自身Haze4K阶段的PSNR下降；低于0表示反而提升。',
              '两组从各自不同的Haze4K模型接续，只评价整段序列效果；不是共同权重的第四阶段消融。',
              '一个配对种子不代表五任务/多种子结论，GOPRO尚未纳入本表。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Validated denoise pair: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
