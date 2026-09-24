import torch
import torch.nn.functional as F

from .freq_decomp import (
    multilevel_dwt,
    num_subbands,
    freq_band_energy_from_bands,
    freq_band_energy_from_image,
)


# ============================================================
# 空间分块
# ============================================================

def split_spatial_blocks(x, grid_h, grid_w):
    """
    将特征图划分为 grid_h x grid_w 个空间块。
    x: [B, C, H, W]
    返回: list of [B, C, h, w]
    """
    B, C, H, W = x.shape
    h = H // grid_h
    w = W // grid_w
    blocks = []
    for i in range(grid_h):
        for j in range(grid_w):
            block = x[:, :, i*h:(i+1)*h, j*w:(j+1)*w]
            blocks.append(block)
    return blocks


# ============================================================
# 基础 im2col
# ============================================================

def im2col_block(block, kernel_size, stride=1, padding=None):
    """
    对空间块做 im2col。
    block: [B, C, h, w]
    返回: [B, C*K*K, L]
    """
    if padding is None:
        padding = kernel_size // 2
    return F.unfold(block, kernel_size=kernel_size, stride=stride, padding=padding)


def spatial_aware_subsample(H, ratio=0.1, energy_metric='l2'):
    """
    空间感知子采样：从 H 的列中选取能量较高的部分。
    H: [d, n] 每一列是一个 patch 向量
    返回: [d, n_sub]
    """
    d, n = H.shape
    n_sub = max(1, int(n * ratio))
    if n_sub >= n:
        return H
    if energy_metric == 'l2':
        energy = H.pow(2).sum(dim=0)  # [n]
    elif energy_metric == 'grad':
        energy = (H[:, 1:] - H[:, :-1]).pow(2).sum(dim=0)
        energy = torch.cat([energy[:1], energy], dim=0)
    else:
        energy = H.abs().sum(dim=0)
    idx = torch.topk(energy, n_sub, largest=True).indices
    idx = torch.sort(idx).values
    return H[:, idx]


# ============================================================
# 按子带 im2col
# ============================================================

def im2col_subbands(x, kernel_size, stride=1, padding=None, level=2):
    """
    对输入 x 做多级 DWT，再对每个子带做 im2col。
    x: [B, C, H, W]
    返回: list of [d, n_sub]，长度 = 子带数（level=2 时为 7）
    """
    if padding is None:
        padding = kernel_size // 2

    bands = multilevel_dwt(x, level=level)
    results = []
    for b in bands:
        H = F.unfold(b, kernel_size=kernel_size, stride=stride, padding=padding)
        # H: [B, C*K*K, L] -> [d, B*L]
        B_, d_, L_ = H.shape
        H = H.reshape(B_, d_, L_).permute(1, 0, 2).reshape(d_, B_ * L_)
        results.append(H)
    return results


def spatial_subband_im2col(
    x,
    grid_h,
    grid_w,
    kernel_size,
    stride=1,
    padding=None,
    level=2,
    sample_ratio=0.1,
):
    """
    统一入口：
      1. 空间分块
      2. 每块做多级 DWT + im2col
      3. 每子带做空间感知子采样
    返回: list of (g, f, H_sub)
        g: 空间块索引
        f: 子带索引
        H_sub: [d, n_sub]
    """
    if padding is None:
        padding = kernel_size // 2

    blocks = split_spatial_blocks(x, grid_h, grid_w)
    results = []
    for g, block in enumerate(blocks):
        subbands = im2col_subbands(
            block, kernel_size, stride=stride, padding=padding, level=level
        )
        for f, H_sub in enumerate(subbands):
            H_sub = spatial_aware_subsample(H_sub, ratio=sample_ratio)
            results.append((g, f, H_sub))
    return results


# ============================================================
# 导出
# ============================================================

__all__ = [
    "split_spatial_blocks",
    "im2col_block",
    "spatial_aware_subsample",
    "im2col_subbands",
    "spatial_subband_im2col",
    "num_subbands",
    "freq_band_energy_from_bands",
    "freq_band_energy_from_image",
]