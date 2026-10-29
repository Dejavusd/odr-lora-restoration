import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .transforms import Compose, RandomCrop, RandomFlip, RandomRotate, ToTensor


def _extract_id(filename):
    """从文件名中提取数字 ID，用于配对。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    prefix = base.split('_')[0]
    if prefix.isdigit():
        return str(int(prefix))
    nums = re.findall(r'\d+', base)
    if nums:
        return str(int(nums[-1]))
    return base


def pair_images(input_dir, gt_dir, input_pattern="*", gt_pattern="*"):
    """
    将输入图像和 GT 图像配对。
    支持：
      1. input_dir 下存在 input/ 和 gt/ 子目录；
      2. 优先按完整文件名、其次按唯一文件主名配对；
      3. 名称不同时按数字 ID 配对，但每个 ID 必须只有一张 GT。
    允许多个退化输入对应同一 GT；拒绝歧义、漏配和按排序猜测。
    返回 (input_paths, gt_paths)。
    """
    if os.path.abspath(input_dir) == os.path.abspath(gt_dir):
        root = input_dir
        gt_dir = os.path.join(root, 'gt')
        for folder in ('input', 'haze'):
            candidate = os.path.join(root, folder)
            if os.path.isdir(candidate):
                input_dir = candidate
                break
        else:
            raise RuntimeError(f'找不到 input/ 或 haze/：{root}')
        if not os.path.isdir(gt_dir):
            raise RuntimeError(f'找不到 gt/：{root}')

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def image_files(directory, pattern):
        return sorted(
            p for p in glob.glob(os.path.join(directory, pattern))
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in extensions
        )

    input_paths = image_files(input_dir, input_pattern)
    gt_paths = image_files(gt_dir, gt_pattern)

    if len(input_paths) == 0 or len(gt_paths) == 0:
        raise RuntimeError(
            f"未找到图像：input_dir={input_dir}, gt_dir={gt_dir}, "
            f"patterns=({input_pattern}, {gt_pattern})"
        )

    if os.path.abspath(input_dir) == os.path.abspath(gt_dir):
        if len(input_paths) == len(gt_paths):
            return input_paths, gt_paths
        else:
            raise RuntimeError(
                f"同一目录下输入/GT 数量不一致：{len(input_paths)} vs {len(gt_paths)}"
            )

    gt_names = {os.path.basename(p): p for p in gt_paths}
    gt_stems, gt_map = {}, {}
    for p in gt_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        gt_stems.setdefault(stem, []).append(p)
        idx = _extract_id(p)
        gt_map.setdefault(idx, []).append(p)

    paired_gt = []
    for path in input_paths:
        name = os.path.basename(path)
        if name in gt_names:
            paired_gt.append(gt_names[name])
            continue
        candidates = gt_stems.get(os.path.splitext(name)[0], [])
        if not candidates:
            candidates = gt_map.get(_extract_id(path), [])
        if len(candidates) != 1:
            raise RuntimeError(
                f'图像配对缺失或存在歧义：{path}，GT 候选={candidates[:5]}。'
                '请使用一一对应的完整文件名，不能按排序猜测。'
            )
        paired_gt.append(candidates[0])
    unused = sorted(set(gt_paths) - set(paired_gt))
    if unused:
        raise RuntimeError(
            f'存在 {len(unused)} 张未配对 GT，例如 {unused[:5]}，请检查文件名或过滤规则。'
        )
    return input_paths, paired_gt


def pair_gopro_sequences(split_dir, input_folder='blur'):
    """Pair exact filenames within each sequence; frame IDs are not global."""
    if input_folder not in ('blur', 'blur_gamma'):
        raise ValueError('GOPRO input_folder 必须是 blur 或 blur_gamma')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'GOPRO 目录不存在：{split_dir}')
    inputs, targets = [], []
    for name in sorted(os.listdir(split_dir)):
        sequence = os.path.join(split_dir, name)
        if not os.path.isdir(sequence):
            continue
        input_dir = os.path.join(sequence, input_folder)
        gt_dir = os.path.join(sequence, 'sharp')
        if not os.path.isdir(input_dir) or not os.path.isdir(gt_dir):
            raise RuntimeError(f'GOPRO 序列缺少 {input_folder}/sharp：{sequence}')

        def images(directory):
            return {f: os.path.join(directory, f) for f in os.listdir(directory)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and os.path.isfile(os.path.join(directory, f))}

        blurred, sharp = images(input_dir), images(gt_dir)
        if not blurred or blurred.keys() != sharp.keys():
            raise RuntimeError(
                f'GOPRO 序列配对不完整：{sequence}，'
                f'缺少 GT={sorted(blurred.keys() - sharp.keys())[:5]}，'
                f'缺少输入={sorted(sharp.keys() - blurred.keys())[:5]}'
            )
        for filename in sorted(blurred):
            inputs.append(blurred[filename])
            targets.append(sharp[filename])
    if not inputs:
        raise RuntimeError(f'GOPRO 没有可用序列：{split_dir}')
    return inputs, targets


class RestorationDataset(Dataset):
    """
    成对图像复原数据集。
    """
    def __init__(
        self,
        input_paths,
        gt_paths,
        patch_size=256,
        augment=True,
        normalize=False,
        strict_size=False,
    ):
        assert len(input_paths) == len(gt_paths), \
            f"输入/GT 数量不一致：{len(input_paths)} vs {len(gt_paths)}"
        self.input_paths = input_paths
        self.gt_paths = gt_paths
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        self.strict_size = strict_size

        if augment:
            self.transform = Compose([
                RandomCrop(patch_size),
                RandomFlip(p=0.5),
                RandomRotate(p=0.5),
                ToTensor(),
            ])
        else:
            self.transform = Compose([
                ToTensor(),
            ])

    def __len__(self):
        return len(self.input_paths)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        inp = self._load_image(self.input_paths[idx])
        gt = self._load_image(self.gt_paths[idx])

        if inp.size != gt.size:
            if self.strict_size:
                raise ValueError(f'输入/GT 尺寸不一致：{self.input_paths[idx]} {inp.size}，'
                                 f'{self.gt_paths[idx]} {gt.size}')
            gt = gt.resize(inp.size, Image.BICUBIC)

        inp, gt = self.transform(inp, gt)

        if self.normalize:
            inp = inp * 2 - 1
            gt = gt * 2 - 1

        return {
            "input": inp,
            "gt": gt,
            "input_path": self.input_paths[idx],
            "gt_path": self.gt_paths[idx],
        }


class GaussianDenoisingDataset(Dataset):
    """Clean images with online AWGN; evaluation noise is fixed per image.

    Grayscale images receive one noise channel replicated across RGB. Noisy
    inputs are not clipped, preserving the specified Gaussian distribution.
    """
    def __init__(self, directory, sigma=25, patch_size=128, train=True,
                 seed=42, dataset_name='denoise'):
        self.paths = sorted(
            os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
            and os.path.splitext(name)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        )
        if not self.paths:
            raise RuntimeError(f'没有找到干净图像：{directory}')
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise ValueError('noise_sigma 必须非负')
        self.train, self.seed, self.dataset_name = train, int(seed), dataset_name
        transforms = [RandomCrop(patch_size), RandomFlip(), RandomRotate()] if train else []
        self.transform = Compose(transforms + [ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as image:
            grayscale = image.mode in ('1', 'L', 'I', 'F', 'I;16')
            clean, _ = self.transform(image.convert('RGB'), None)
        generator = None if self.train else torch.Generator().manual_seed(self.seed + idx)
        shape = (1, *clean.shape[1:]) if grayscale else clean.shape
        noise = torch.randn(shape, generator=generator) * (self.sigma / 255.0)
        return {'input': clean + noise, 'gt': clean, 'input_path': path,
                'gt_path': path, 'dataset_name': self.dataset_name,
                'grayscale': grayscale}


def build_task_loaders(cfg, task_name):
    """
    根据配置和任务名构建训练/验证 DataLoader。
    """
    root = cfg["data"]["root"]
    task_cfg = None
    for t in cfg["data"]["tasks"]:
        if t["name"] == task_name:
            task_cfg = t
            break
    if task_cfg is None:
        raise ValueError(f"未知任务：{task_name}")

    dataset_name = task_cfg["dataset"]
    train_split = task_cfg.get("train_split", "train")
    val_split = task_cfg.get("val_split", "test")
    patch_size = task_cfg.get("patch_size", 256)
    batch_size = task_cfg.get("batch_size", 8)

    train_dir = os.path.join(root, dataset_name, train_split)
    val_dir = os.path.join(root, dataset_name, val_split)

    if "train_dir" in task_cfg:
        train_dir = os.path.join(root, task_cfg["train_dir"])
    if "val_dir" in task_cfg:
        val_dir = os.path.join(root, task_cfg["val_dir"])

    if task_cfg.get('type') == 'gaussian_denoising':
        sigma = task_cfg.get('noise_sigma', 25)
        seed = task_cfg.get('noise_seed', 42)
        train_set = GaussianDenoisingDataset(
            train_dir, sigma, patch_size, train=True, seed=seed)
        val_sets = task_cfg.get('val_sets', [
            {'name': dataset_name, 'dir': os.path.relpath(val_dir, root)}])
        val_set = ConcatDataset([
            GaussianDenoisingDataset(os.path.join(root, item['dir']), sigma,
                                     train=False, seed=seed, dataset_name=item['name'])
            for item in val_sets
        ])
        workers = task_cfg.get('num_workers', 4)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=False),
            DataLoader(val_set, batch_size=1, shuffle=False,
                       num_workers=workers, pin_memory=True),
        )

    input_pattern = task_cfg.get("input_pattern", "*")
    gt_pattern = task_cfg.get("gt_pattern", "*")

    if task_cfg.get('type') == 'gopro':
        folder = task_cfg.get('input_folder', 'blur')
        train_inputs, train_gts = pair_gopro_sequences(train_dir, folder)
        val_inputs, val_gts = pair_gopro_sequences(val_dir, folder)
    else:
        train_inputs, train_gts = pair_images(
            train_dir, train_dir, input_pattern=input_pattern, gt_pattern=gt_pattern)
        val_inputs, val_gts = pair_images(
            val_dir, val_dir, input_pattern=input_pattern, gt_pattern=gt_pattern)

    train_set = RestorationDataset(
        train_inputs, train_gts,
        patch_size=patch_size,
        augment=True,
        normalize=False,
    )
    val_set = RestorationDataset(
        val_inputs, val_gts,
        patch_size=patch_size,
        augment=False,
        normalize=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
