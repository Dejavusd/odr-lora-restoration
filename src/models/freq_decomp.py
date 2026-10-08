import torch
import torch.nn.functional as F


# ============================================================
# 频带定义
# ============================================================

FREQ_BANDS = ['LL2', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1']


def num_subbands(level=2):
    """
    level=1 -> 4 个子带：LL1, LH1, HL1, HH1
    level=2 -> 7 个子带：LL2, LH2, HL2, HH2, LH1, HL1, HH1
    """
    if level == 1:
        return 4
    elif level == 2:
        return 7
    else:
        return 1 + 3 * level


# ============================================================
# Haar 小波变换
# ============================================================

def _pad_to_even(x):
    """如果 H 或 W 是奇数，pad 1 像素使其变为偶数。"""
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, (pad_h, pad_w)


def _unpad(x, pad):
    """去掉之前 pad 的像素。"""
    pad_h, pad_w = pad
    if pad_h == 0 and pad_w == 0:
        return x
    B, C, H, W = x.shape
    return x[:, :, :H - pad_h, :W - pad_w]


def haar_dwt2(x):
    """
    一级 Haar 二维 DWT。
    x: [B, C, H, W]
    返回 LL, LH, HL, HH，每个 [B, C, H/2, W/2]
    """
    x, pad = _pad_to_even(x)
    # 行方向
    x0 = x[:, :, 0::2, :]
    x1 = x[:, :, 1::2, :]
    # 列方向
    x00 = x0[:, :, :, 0::2]
    x01 = x0[:, :, :, 1::2]
    x10 = x1[:, :, :, 0::2]
    x11 = x1[:, :, :, 1::2]

    LL = (x00 + x01 + x10 + x11) / 2.0
    LH = (x00 - x01 + x10 - x11) / 2.0
    HL = (x00 + x01 - x10 - x11) / 2.0
    HH = (x00 - x01 - x10 + x11) / 2.0
    return LL, LH, HL, HH


def haar_iwt2(LL, LH, HL, HH, pad=(0, 0)):
    """
    一级 Haar 二维逆 DWT。
    LL, LH, HL, HH: [B, C, H/2, W/2]
    返回 [B, C, H, W]
    """
    B, C, H, W = LL.shape
    x = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = (LL + LH + HL + HH) / 2.0
    x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) / 2.0
    x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) / 2.0
    x[:, :, 1::2, 1::2] = (LL - LH - HL + HH) / 2.0
    return _unpad(x, pad)


def multilevel_dwt(x, level=2):
    """
    多级 Haar DWT。
    level=1: 返回 [LL1, LH1, HL1, HH1]
    level=2: 返回 [LL2, LH2, HL2, HH2, LH1, HL1, HH1]
    """
    x, pad1 = _pad_to_even(x)
    LL1, LH1, HL1, HH1 = haar_dwt2(x)
    if level == 1:
        return [LL1, LH1, HL1, HH1]

    LL2, LH2, HL2, HH2 = haar_dwt2(LL1)
    return [LL2, LH2, HL2, HH2, LH1, HL1, HH1]


def multilevel_iwt(bands, level=2):
    """
    多级 Haar 逆 DWT。
    level=1: bands = [LL1, LH1, HL1, HH1]
    level=2: bands = [LL2, LH2, HL2, HH2, LH1, HL1, HH1]
    """
    if level == 1:
        LL1, LH1, HL1, HH1 = bands
        return haar_iwt2(LL1, LH1, HL1, HH1)

    LL2, LH2, HL2, HH2, LH1, HL1, HH1 = bands
    LL1 = haar_iwt2(LL2, LH2, HL2, HH2)
    return haar_iwt2(LL1, LH1, HL1, HH1)


# ============================================================
# 频带能量统计
# ============================================================

def freq_band_energy_from_bands(bands):
    """给定子带列表，返回各子带能量占比。"""
    energies = [b.pow(2).sum().item() for b in bands]
    total = sum(energies) + 1e-12
    return [e / total for e in energies]


def freq_band_energy_from_image(x, level=2):
    """直接从图像/特征图计算各 DWT 子带能量占比。"""
    bands = multilevel_dwt(x, level=level)
    return freq_band_energy_from_bands(bands)


# ============================================================
# DCT 备用（保留，但默认不用）
# ============================================================

def dct_2d(x):
    """
    简易二维 DCT-II 变换（用 FFT 近似），输入 [B, C, H, W]。
    实际工程中可用 scipy.fft.dctn。
    """
    X = torch.fft.fft2(x, norm='ortho')
    return X


# ============================================================
# 兼容旧接口
# ============================================================

def freq_band_split(H, bands=7, dim=0):
    """
    频带划分。
    - 如果 H 是 4D 图像张量 [B, C, H, W]，走 DWT 子带，返回 list of tensor。
    - 如果 H 是 2D 矩阵 [d, n]（im2col 后），退化为按元素索引分段。
    """
    if H.dim() == 4:
        # 图像张量：使用多级 DWT
        level = 2 if bands >= 7 else 1
        return multilevel_dwt(H, level=level)
    else:
        # 旧行为：按元素索引分段
        d, n = H.shape
        seg = d // bands
        parts = []
        for b in range(bands):
            start = b * seg
            end = d if b == bands - 1 else (b + 1) * seg
            parts.append(H[start:end, :])
        return parts


def freq_band_merge(parts):
    """将频带分段合并回原维度（仅适用于按索引分段的情况）。"""
    return torch.cat(parts, dim=0)


def freq_energy_ratio(H, bands=7):
    """
    频带能量占比。
    - 如果 H 是 4D，走 DWT 子带能量。
    - 如果 H 是 2D，按索引分段后统计。
    """
    if H.dim() == 4:
        level = 2 if bands >= 7 else 1
        return freq_band_energy_from_image(H, level=level)
    else:
        parts = freq_band_split(H, bands=bands)
        energies = [p.pow(2).sum().item() for p in parts]
        total = sum(energies) + 1e-12
        return [e / total for e in energies]


# ============================================================
# 导出
# ============================================================

__all__ = [
    "FREQ_BANDS",
    "num_subbands",
    "haar_dwt2",
    "haar_iwt2",
    "multilevel_dwt",
    "multilevel_iwt",
    "freq_band_energy_from_bands",
    "freq_band_energy_from_image",
    "dct_2d",
    "freq_band_split",
    "freq_band_merge",
    "freq_energy_ratio",
]