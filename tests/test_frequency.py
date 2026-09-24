import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.freq_decomp import (
    FREQ_BANDS,
    num_subbands,
    haar_dwt2,
    haar_iwt2,
    multilevel_dwt,
    multilevel_iwt,
    freq_band_energy_from_bands,
    freq_band_energy_from_image,
)
from src.models.spatial_block import (
    im2col_subbands,
    spatial_subband_im2col,
    split_spatial_blocks,
)
from src.losses.losses import (
    FrequencyComplementarityLoss,
    FrequencySelectionRegularization,
    compute_frequency_loss,
    compute_frequency_selection_loss,
)


# ============================================================
# DWT / IWT
# ============================================================

def test_haar_dwt_iwt_reconstruction():
    """测试一级 Haar DWT + IWT 可逆。"""
    x = torch.randn(2, 3, 32, 32)
    LL, LH, HL, HH = haar_dwt2(x)
    x_rec = haar_iwt2(LL, LH, HL, HH)
    assert x_rec.shape == x.shape, f"重建形状不一致: {x_rec.shape} vs {x.shape}"
    assert torch.allclose(x, x_rec, atol=1e-5), \
        f"一级 DWT/IWT 不可逆，最大误差: {(x - x_rec).abs().max().item()}"
    print("test_haar_dwt_iwt_reconstruction 通过")


def test_multilevel_dwt_iwt_reconstruction():
    """测试二级 Haar DWT + IWT 可逆。"""
    x = torch.randn(2, 3, 64, 64)
    bands = multilevel_dwt(x, level=2)
    x_rec = multilevel_iwt(bands, level=2)
    assert x_rec.shape == x.shape, f"重建形状不一致: {x_rec.shape} vs {x.shape}"
    assert torch.allclose(x, x_rec, atol=1e-5), \
        f"二级 DWT/IWT 不可逆，最大误差: {(x - x_rec).abs().max().item()}"
    print("test_multilevel_dwt_iwt_reconstruction 通过")


def test_subband_count():
    """测试子带数量与 level 对应。"""
    x = torch.randn(2, 3, 64, 64)

    bands1 = multilevel_dwt(x, level=1)
    assert len(bands1) == 4, f"level=1 应返回 4 子带，实际 {len(bands1)}"
    assert num_subbands(level=1) == 4

    bands2 = multilevel_dwt(x, level=2)
    assert len(bands2) == 7, f"level=2 应返回 7 子带，实际 {len(bands2)}"
    assert num_subbands(level=2) == 7

    # 检查 FREQ_BANDS 常量
    assert len(FREQ_BANDS) == 7
    print("test_subband_count 通过")


def test_subband_shapes():
    """测试多级 DWT 后各子带的空间尺寸。"""
    x = torch.randn(2, 3, 64, 64)
    bands = multilevel_dwt(x, level=2)
    # LL2, LH2, HL2, HH2: 16x16; LH1, HL1, HH1: 32x32
    expected_sizes = [(16, 16), (16, 16), (16, 16), (16, 16),
                      (32, 32), (32, 32), (32, 32)]
    for i, (b, (h, w)) in enumerate(zip(bands, expected_sizes)):
        assert b.shape[-2:] == (h, w), \
            f"子带 {FREQ_BANDS[i]} 尺寸应为 {(h, w)}，实际 {b.shape[-2:]}"
    print("test_subband_shapes 通过")


def test_odd_size_input():
    """测试奇数尺寸输入时 DWT/IWT 仍能可逆。"""
    x = torch.randn(2, 3, 31, 33)
    bands = multilevel_dwt(x, level=2)
    x_rec = multilevel_iwt(bands, level=2)
    # 逆变换后尺寸可能被裁剪回原尺寸
    assert x_rec.shape == x.shape, \
        f"奇数尺寸重建形状不一致: {x_rec.shape} vs {x.shape}"
    assert torch.allclose(x, x_rec, atol=1e-5), \
        f"奇数尺寸 DWT/IWT 不可逆，最大误差: {(x - x_rec).abs().max().item()}"
    print("test_odd_size_input 通过")


def test_small_size_input():
    """测试极小尺寸输入。"""
    x = torch.randn(1, 1, 8, 8)
    bands = multilevel_dwt(x, level=2)
    x_rec = multilevel_iwt(bands, level=2)
    assert x_rec.shape == x.shape
    assert torch.allclose(x, x_rec, atol=1e-5)
    print("test_small_size_input 通过")


# ============================================================
# 频带能量统计
# ============================================================

def test_freq_band_energy_from_bands():
    """测试从子带列表计算能量占比。"""
    x = torch.randn(2, 3, 64, 64)
    bands = multilevel_dwt(x, level=2)
    ratio = freq_band_energy_from_bands(bands)

    assert len(ratio) == 7, f"能量占比长度应为 7，实际 {len(ratio)}"
    assert abs(sum(ratio) - 1.0) < 1e-5, \
        f"能量占比之和应为 1，实际 {sum(ratio)}"
    assert all(r >= 0 for r in ratio), "能量占比不应为负"
    print(f"test_freq_band_energy_from_bands 通过，能量占比: "
          f"{[f'{r:.3f}' for r in ratio]}")


def test_freq_band_energy_from_image():
    """测试直接从图像计算能量占比。"""
    x = torch.randn(2, 3, 64, 64)
    ratio = freq_band_energy_from_image(x, level=2)
    assert len(ratio) == 7
    assert abs(sum(ratio) - 1.0) < 1e-5
    print("test_freq_band_energy_from_image 通过")


def test_energy_ratio_different_inputs():
    """测试不同输入的能量分布不同。"""
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64) * 10.0
    r1 = freq_band_energy_from_image(x1, level=2)
    r2 = freq_band_energy_from_image(x2, level=2)
    # 缩放输入不应改变能量占比（因为归一化了）
    assert all(abs(a - b) < 1e-5 for a, b in zip(r1, r2)), \
        "缩放不应改变能量占比"
    print("test_energy_ratio_different_inputs 通过")


# ============================================================
# im2col_subbands / spatial_subband_im2col
# ============================================================

def test_im2col_subbands():
    """测试按子带 im2col 的输出结构。"""
    x = torch.randn(2, 4, 32, 32)
    kernel_size = 3
    results = im2col_subbands(x, kernel_size, stride=1, padding=1, level=2)

    assert len(results) == 7, f"应返回 7 个子带，实际 {len(results)}"
    for i, H in enumerate(results):
        assert H.dim() == 2, f"子带 {i} 应为 2D 矩阵，实际 {H.dim()}D"
        d = H.shape[0]
        assert d == 4 * kernel_size * kernel_size, \
            f"子带 {i} 的 d 应为 {4 * kernel_size * kernel_size}，实际 {d}"
    print(f"test_im2col_subbands 通过，子带数: {len(results)}")


def test_im2col_subbands_level1():
    """测试 level=1 时 im2col 返回 4 个子带。"""
    x = torch.randn(2, 4, 32, 32)
    results = im2col_subbands(x, 3, stride=1, padding=1, level=1)
    assert len(results) == 4, f"level=1 应返回 4 个子带，实际 {len(results)}"
    print("test_im2col_subbands_level1 通过")


def test_spatial_subband_im2col():
    """测试空间分块 + 子带 im2col 的输出格式。"""
    x = torch.randn(2, 4, 32, 32)
    grid_h, grid_w = 2, 2
    kernel_size = 3
    results = spatial_subband_im2col(
        x, grid_h, grid_w, kernel_size,
        stride=1, padding=1, level=2, sample_ratio=0.5,
    )

    # 应为 4 块 * 7 子带 = 28 个 (g, f, H)
    assert len(results) == grid_h * grid_w * 7, \
        f"应返回 {grid_h * grid_w * 7} 项，实际 {len(results)}"

    for g, f, H in results:
        assert 0 <= g < grid_h * grid_w, f"空间块索引越界: {g}"
        assert 0 <= f < 7, f"子带索引越界: {f}"
        assert H.dim() == 2, f"H 应为 2D，实际 {H.dim()}D"
    print(f"test_spatial_subband_im2col 通过，返回 {len(results)} 项")


def test_split_spatial_blocks():
    """测试空间分块。"""
    x = torch.randn(2, 4, 32, 32)
    blocks = split_spatial_blocks(x, 2, 2)
    assert len(blocks) == 4, f"应分成 4 块，实际 {len(blocks)}"
    for b in blocks:
        assert b.shape == (2, 4, 16, 16), f"块形状错误: {b.shape}"
    print("test_split_spatial_blocks 通过")


# ============================================================
# 频带损失
# ============================================================

def test_frequency_complementarity_loss():
    """测试频带互补损失。"""
    loss_fn = FrequencyComplementarityLoss()

    # 完全重叠
    p_t = torch.tensor([0.5, 0.5])
    p_old = torch.tensor([0.5, 0.5])
    loss_same = loss_fn(p_t, p_old)

    # 完全不重叠
    p_old2 = torch.tensor([0.0, 1.0])
    loss_diff = loss_fn(p_t, p_old2)

    assert loss_same > loss_diff, \
        f"完全重叠损失应大于不重叠: {loss_same} vs {loss_diff}"
    print(f"test_frequency_complementarity_loss 通过，"
          f"重叠={loss_same:.4f}, 不重叠={loss_diff:.4f}")


def test_frequency_complementarity_loss_none():
    """测试 p_old 为 None 时返回 0。"""
    loss_fn = FrequencyComplementarityLoss()
    p_t = torch.tensor([0.5, 0.5])
    loss = loss_fn(p_t, None)
    assert loss.item() == 0.0
    print("test_frequency_complementarity_loss_none 通过")


def test_frequency_selection_regularization_entropy():
    """测试频带选择正则化（熵模式）。"""
    reg = FrequencySelectionRegularization(mode='entropy')

    # 均匀分布，熵最大
    uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
    loss_uniform = reg(uniform)

    # 集中分布，熵较小
    concentrated = torch.tensor([0.9, 0.05, 0.03, 0.02])
    loss_concentrated = reg(concentrated)

    assert loss_concentrated > loss_uniform, \
        f"集中分布应受更大惩罚: {loss_concentrated} vs {loss_uniform}"
    print(f"test_frequency_selection_regularization_entropy 通过，"
          f"均匀={loss_uniform:.4f}, 集中={loss_concentrated:.4f}")


def test_frequency_selection_regularization_empty():
    """测试空输入。"""
    reg = FrequencySelectionRegularization(mode='entropy')
    loss = reg(None)
    assert loss.item() == 0.0
    loss = reg(torch.tensor([]))
    assert loss.item() == 0.0
    print("test_frequency_selection_regularization_empty 通过")


# ============================================================
# compute_frequency_loss / compute_frequency_selection_loss
# ============================================================

def test_compute_frequency_loss_no_stats():
    """测试模型未保存频带统计时返回 0。"""
    import torch.nn as nn

    model = nn.Sequential(nn.Conv2d(3, 3, 3))
    loss = compute_frequency_loss(model, task_idx=1, freq_bands=7)
    assert loss.item() == 0.0
    print("test_compute_frequency_loss_no_stats 通过")


def test_compute_frequency_loss_with_stats():
    """测试模型保存了频带统计时的损失计算。"""
    import torch.nn as nn

    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_energy_current = torch.tensor([0.1, 0.2, 0.3, 0.4])
            self.freq_energy_old = torch.tensor([0.4, 0.3, 0.2, 0.1])

    model = nn.Sequential(DummyModule())
    loss = compute_frequency_loss(model, task_idx=1, freq_bands=4)
    # 重叠度为 min(p_t, p_old) 之和
    expected = min(0.1, 0.4) + min(0.2, 0.3) + min(0.3, 0.2) + min(0.4, 0.1)
    assert abs(loss.item() - expected) < 1e-5, \
        f"损失应为 {expected}，实际 {loss.item()}"
    print(f"test_compute_frequency_loss_with_stats 通过，损失: {loss.item():.4f}")


def test_compute_frequency_selection_loss():
    """测试频带选择损失。"""
    import torch.nn as nn

    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_importance = torch.tensor([10.0, 5.0, 3.0, 2.0])

    model = nn.Sequential(DummyModule())
    loss = compute_frequency_selection_loss(model)
    assert loss.item() != 0.0, "频带选择损失不应为 0"
    print(f"test_compute_frequency_selection_loss 通过，损失: {loss.item():.4f}")


def test_compute_frequency_selection_loss_no_stats():
    """测试模型未保存频带重要性时返回 0。"""
    import torch.nn as nn

    model = nn.Sequential(nn.Conv2d(3, 3, 3))
    loss = compute_frequency_selection_loss(model)
    assert loss.item() == 0.0
    print("test_compute_frequency_selection_loss_no_stats 通过")


if __name__ == '__main__':
    # DWT / IWT
    test_haar_dwt_iwt_reconstruction()
    test_multilevel_dwt_iwt_reconstruction()
    test_subband_count()
    test_subband_shapes()
    test_odd_size_input()
    test_small_size_input()

    # 能量统计
    test_freq_band_energy_from_bands()
    test_freq_band_energy_from_image()
    test_energy_ratio_different_inputs()

    # im2col
    test_im2col_subbands()
    test_im2col_subbands_level1()
    test_spatial_subband_im2col()
    test_split_spatial_blocks()

    # 频带损失
    test_frequency_complementarity_loss()
    test_frequency_complementarity_loss_none()
    test_frequency_selection_regularization_entropy()
    test_frequency_selection_regularization_empty()

    # compute 函数
    test_compute_frequency_loss_no_stats()
    test_compute_frequency_loss_with_stats()
    test_compute_frequency_selection_loss()
    test_compute_frequency_selection_loss_no_stats()

    print("所有频域测试通过。")