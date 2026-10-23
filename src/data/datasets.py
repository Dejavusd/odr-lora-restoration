import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import Compose, RandomCrop, RandomFlip, RandomRotate, ToTensor


def _extract_id(filename):
    """从文件名中提取数字 ID，用于配对。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r'\d+', base)
    if nums:
        return nums[-1]
    return base


def pair_images(input_dir, gt_dir, input_pattern="*", gt_pattern="*"):
    """
    将输入图像和 GT 图像配对。
    支持：
      1. input_dir 下存在 input/ 和 gt/ 子目录；
      2. input_dir 和 gt_dir 为不同目录，按文件名中的数字 ID 配对；
      3. 按排序后一一对应（数量相同）。
    返回 (input_paths, gt_paths)。
    """
    sub_input = os.path.join(input_dir, "input")
    sub_gt = os.path.join(input_dir, "gt")
    if os.path.isdir(sub_input) and os.path.isdir(sub_gt):
        input_dir = sub_input
        gt_dir = sub_gt

    input_paths = sorted(glob.glob(os.path.join(input_dir, input_pattern)))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, gt_pattern)))

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

    input_map = {}
    for p in input_paths:
        idx = _extract_id(p)
        input_map.setdefault(idx, []).append(p)
    gt_map = {}
    for p in gt_paths:
        idx = _extract_id(p)
        gt_map.setdefault(idx, []).append(p)

    common_ids = sorted(set(input_map.keys()) & set(gt_map.keys()))
    if len(common_ids) > 0:
        paired_input, paired_gt = [], []
        for idx in common_ids:
            for ip in input_map[idx]:
                for gp in gt_map[idx]:
                    paired_input.append(ip)
                    paired_gt.append(gp)
        if len(paired_input) > 0:
            return paired_input, paired_gt

    if len(input_paths) != len(gt_paths):
        raise RuntimeError(
            f"无法配对：输入 {len(input_paths)} 张，GT {len(gt_paths)} 张。"
            f"请检查文件名或目录结构。"
        )
    return input_paths, gt_paths


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
    ):
        assert len(input_paths) == len(gt_paths), \
            f"输入/GT 数量不一致：{len(input_paths)} vs {len(gt_paths)}"
        self.input_paths = input_paths
        self.gt_paths = gt_paths
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize

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

    input_pattern = task_cfg.get("input_pattern", "*")
    gt_pattern = task_cfg.get("gt_pattern", "*")

    train_inputs, train_gts = pair_images(
        train_dir, train_dir,
        input_pattern=input_pattern,
        gt_pattern=gt_pattern,
    )
    val_inputs, val_gts = pair_images(
        val_dir, val_dir,
        input_pattern=input_pattern,
        gt_pattern=gt_pattern,
    )

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