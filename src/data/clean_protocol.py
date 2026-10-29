"""Immutable train/validation/test manifests; raw data are never moved."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import DataLoader

from src.data.datasets import pair_images, RestorationDataset
from src.utils.training_state import atomic_json


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record_pair(pair):
    inp, gt = map(Path, pair)
    before = [[p.stat().st_size, p.stat().st_mtime_ns] for p in (inp, gt)]
    input_bytes, gt_bytes = inp.read_bytes(), gt.read_bytes()
    with Image.open(io.BytesIO(input_bytes)) as x, Image.open(io.BytesIO(gt_bytes)) as y:
        if x.size != y.size:
            raise ValueError(f'输入/GT 尺寸不一致：{inp}, {gt}')
        pixel_hash = hashlib.sha256(str(y.size).encode() + y.convert('RGB').tobytes()).hexdigest()
        size = list(y.size)
    after = [[p.stat().st_size, p.stat().st_mtime_ns] for p in (inp, gt)]
    if before != after:
        raise ValueError(f'审计期间文件发生改变：{inp}, {gt}')
    return dict(input=str(inp.resolve()), gt=str(gt.resolve()), size=size,
                gt_group=pixel_hash, input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                gt_sha256=hashlib.sha256(gt_bytes).hexdigest(),
                file_stats=after)


def group_indices(records, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction 必须在 (0,1)')
    groups = sorted({r['gt_group'] for r in records})
    if len(groups) < 2:
        raise ValueError('至少需要两个不同 GT 组')
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups)-1, round(len(groups)*fraction)))
    heldout = set(groups[:count])
    return ([i for i, r in enumerate(records) if r['gt_group'] not in heldout],
            [i for i, r in enumerate(records) if r['gt_group'] in heldout])


def prepare_splits(cfg, workers=4):
    destination = Path(cfg['data']['split_manifest'])
    if destination.exists():
        raise FileExistsError(f'划分文件已存在，不覆盖：{destination}')
    root = Path(cfg['data']['root']).resolve()
    manifest = dict(version=1, root=str(root), tasks={},
        split_seed=cfg['data'].get('split_seed', 20261029),
        validation_fraction=cfg['data'].get('validation_fraction', .1),
        protocol='Exact filenames first; decoded GT groups; fail on train/test GT overlap; '
                 'validation from TRAIN only; no raw image mutation. Near-duplicate scenes not guaranteed disjoint.')
    for task in cfg['data']['tasks']:
        rows = {}
        for split in ('train', 'test'):
            directory = root / task['dataset'] / task.get(f'{split}_split', split)
            a, b = pair_images(str(directory), str(directory))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows[split] = []
                for i, record in enumerate(pool.map(record_pair, zip(a, b)), 1):
                    rows[split].append(record)
                    if i % 500 == 0 or i == len(a):
                        print(f'AUDIT {task["dataset"]} {split}: {i}/{len(a)}', flush=True)
        train_groups = {r['gt_group'] for r in rows['train']}
        test_groups = {r['gt_group'] for r in rows['test']}
        if train_groups & test_groups:
            raise ValueError(f'{task["dataset"]} 训练/测试存在 {len(train_groups & test_groups)} 个相同 GT 组，'
                             '请先处理泄漏；不会默默拆分或修改数据')
        train, val = group_indices(rows['train'], manifest['validation_fraction'], manifest['split_seed'])
        manifest['tasks'][task['name']] = dict(dataset=task['dataset'], train_records=rows['train'],
            test_records=rows['test'], train_indices=train, validation_indices=val,
            total_train_pairs=len(rows['train']), train_pairs=len(train), validation_pairs=len(val),
            test_pairs=len(rows['test']), unique_train_gt=len(train_groups), unique_test_gt=len(test_groups))
        print(f'SPLIT {task["name"]}: train={len(train)}, val={len(val)}, test={len(rows["test"])}', flush=True)
    atomic_json(destination, manifest)
    return manifest


def load_manifest(cfg, verify_hashes=False):
    path = Path(cfg['data']['split_manifest'])
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or manifest['root'] != str(Path(cfg['data']['root']).resolve()):
        raise ValueError('划分版本或数据根目录不匹配')
    if (manifest['split_seed'] != cfg['data'].get('split_seed', 20261029)
            or manifest['validation_fraction'] != cfg['data'].get('validation_fraction', .1)):
        raise ValueError('划分种子或比例与配置不匹配')
    if list(manifest['tasks']) != [t['name'] for t in cfg['data']['tasks']]:
        raise ValueError('划分中的任务顺序与配置不一致')
    for task in cfg['data']['tasks']:
        item = manifest['tasks'][task['name']]
        if item['dataset'] != task['dataset']:
            raise ValueError('数据集名称与划分不一致')
        train, val = item['train_indices'], item['validation_indices']
        if (set(train) & set(val) or len(train)+len(val) != len(item['train_records'])
                or set(train) | set(val) != set(range(len(item['train_records'])))):
            raise ValueError('训练/验证索引不构成完整且无重叠的划分')
        a = {item['train_records'][i]['gt_group'] for i in train}
        b = {item['train_records'][i]['gt_group'] for i in val}
        c = {r['gt_group'] for r in item['test_records']}
        if a & b or (a | b) & c:
            raise ValueError('GT 组跨训练/验证/测试泄漏')
        input_sets = [{item['train_records'][i]['input_sha256'] for i in indices} for indices in (train, val)]
        input_sets.append({r['input_sha256'] for r in item['test_records']})
        if input_sets[0] & input_sets[1] or (input_sets[0] | input_sets[1]) & input_sets[2]:
            raise ValueError('相同输入文件跨训练/验证/测试泄漏')
        for row in item['train_records'] + item['test_records']:
            for kind, stat in zip(('input', 'gt'), row['file_stats']):
                p = Path(row[kind])
                if [p.stat().st_size, p.stat().st_mtime_ns] != stat:
                    raise ValueError(f'划分创建后数据发生改变：{p}')
                if verify_hashes and sha256_file(p) != row[f'{kind}_sha256']:
                    raise ValueError(f'数据 SHA256 不一致：{p}')
    return manifest


def make_loader(cfg, manifest, task_name, role):
    task = next(t for t in cfg['data']['tasks'] if t['name'] == task_name)
    item = manifest['tasks'][task_name]
    if role == 'test':
        records = item['test_records']
    elif role in ('train', 'validation'):
        key = 'train_indices' if role == 'train' else 'validation_indices'
        records = [item['train_records'][i] for i in item[key]]
    else:
        raise ValueError(f'Unknown split role: {role}')
    dataset = RestorationDataset([r['input'] for r in records], [r['gt'] for r in records],
        patch_size=task.get('patch_size', 128), augment=role == 'train', strict_size=True)
    return DataLoader(dataset, batch_size=task.get('batch_size', 8) if role == 'train' else 1,
        shuffle=role == 'train', drop_last=False, num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=cfg['experiment'].get('device', 'cuda').startswith('cuda'))
