import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class Compose:
    """按顺序组合多个变换。"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt=None):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class RandomCrop:
    """对输入和 GT 同步随机裁剪。"""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, gt=None):
        w, h = img.size
        th, tw = self.size
        if w < tw or h < th:
            scale = max(th / h, tw / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            if gt is not None:
                gt = gt.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)
        img = img.crop((x, y, x + tw, y + th))
        if gt is not None:
            gt = gt.crop((x, y, x + tw, y + th))
        return img, gt


class RandomFlip:
    """随机水平/垂直翻转。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt=None):
        if random.random() < self.p:
            img = TF.hflip(img)
            if gt is not None:
                gt = TF.hflip(gt)
        if random.random() < self.p:
            img = TF.vflip(img)
            if gt is not None:
                gt = TF.vflip(gt)
        return img, gt


class RandomRotate:
    """随机旋转 90 度的倍数。"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt=None):
        if random.random() < self.p:
            k = random.randint(0, 3)
            img = TF.rotate(img, 90 * k)
            if gt is not None:
                gt = TF.rotate(gt, 90 * k)
        return img, gt


class ToTensor:
    """PIL Image -> torch.Tensor，范围 [0, 1]。"""
    def __call__(self, img, gt=None):
        img = TF.to_tensor(img)
        if gt is not None:
            gt = TF.to_tensor(gt)
        return img, gt


class Normalize:
    """按均值方差归一化。"""
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt=None):
        img = TF.normalize(img, self.mean, self.std)
        if gt is not None:
            gt = TF.normalize(gt, self.mean, self.std)
        return img, gt