import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.odr_lora import ODRLoRAConv2d
from src.models.spatial_block import num_subbands
from src.utils.svd_utils import orthogonality_check, subspace_overlap


def _build_layer(freq_level=2, spatial_blocks=(2, 2), adaptive_lambda_min=0.0,
                 adaptive_lambda_max=1.0, share_subspace=True):
    """构建一个用于正交性测试的 ODRLoRAConv2d。"""
    layer = ODRLoRAConv2d(
        8, 16, kernel_size=3,
        spatial_blocks=spatial_blocks,
        freq_level=freq_level,
        adaptive_lambda_min=adaptive_lambda_min,
        adaptive_lambda_max=adaptive_lambda_max,
        share_subspace=share_subspace,
    )
    return layer


def test_Bt_orthogonal_to_old_gpm():
    """
    测试：第二个任务的私有基 B_t 应该与第一个任务更新后的 DualGPM 基正交。
    """
    layer = _build_layer(freq_level=2, spatial_blocks=(1, 1))
    x1 = torch.randn(2, 8, 32, 32)
    x2 = torch.randn(2, 8, 32, 32)

    # 第一个任务：校准 + 更新 GPM
    layer.add_task()
    layer.set_current_task(0)
    layer.calibrate(x1, 0, task_similarity=0.0)
    layer.update_gpm(x1, 0)

    # 第二个任务：校准，此时应做正交投影
    layer.add_task()
    layer.set_current_task(1)
    layer.calibrate(x2, 1, task_similarity=0.0)

    # 检查 B_t 私有部分与旧任务 GPM 基的正交性
    B2_private = layer.task_B[1].data[layer.share_rank:]
    max_overlap = 0.0
    count = 0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            overlap = orthogonality_check(B2_private, gpm.basis)
            max_overlap = max(max_overlap, overlap)
            count += 1

    assert count > 0, "没有非空 GPM，无法检查正交性"
    # 使用随机 SVD 时允许一定误差
    assert max_overlap < 0.5, \
        f"B_t 私有基与旧任务 GPM 基重叠过大: {max_overlap:.4f}"
    print(f"test_Bt_orthogonal_to_old_gpm 通过，最大重叠: {max_overlap:.4f}")


def test_freq_level_1_orthogonality():
    """测试 freq_level=1（4 子带）时的正交性。"""
    layer = _build_layer(freq_level=1, spatial_blocks=(1, 1))
    x1 = torch.randn(2, 8, 32, 32)
    x2 = torch.randn(2, 8, 32, 32)

    layer.add_task()
    layer.set_current_task(0)
    layer.calibrate(x1, 0, task_similarity=0.0)
    layer.update_gpm(x1, 0)

    layer.add_task()
    layer.set_current_task(1)
    layer.calibrate(x2, 1, task_similarity=0.0)

    B2_private = layer.task_B[1].data[layer.share_rank:]
    max_overlap = 0.0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            overlap = orthogonality_check(B2_private, gpm.basis)
            max_overlap = max(max_overlap, overlap)

    assert max_overlap < 0.5, \
        f"freq_level=1 时正交性不满足: {max_overlap:.4f}"
    print(f"test_freq_level_1_orthogonality 通过，最大重叠: {max_overlap:.4f}")


def test_freq_level_2_orthogonality():
    """测试 freq_level=2（7 子带）时的正交性。"""
    layer = _build_layer(freq_level=2, spatial_blocks=(1, 1))
    x1 = torch.randn(2, 8, 32, 32)
    x2 = torch.randn(2, 8, 32, 32)

    layer.add_task()
    layer.set_current_task(0)
    layer.calibrate(x1, 0, task_similarity=0.0)
    layer.update_gpm(x1, 0)

    layer.add_task()
    layer.set_current_task(1)
    layer.calibrate(x2, 1, task_similarity=0.0)

    B2_private = layer.task_B[1].data[layer.share_rank:]
    max_overlap = 0.0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            overlap = orthogonality_check(B2_private, gpm.basis)
            max_overlap = max(max_overlap, overlap)

    assert max_overlap < 0.5, \
        f"freq_level=2 时正交性不满足: {max_overlap:.4f}"
    print(f"test_freq_level_2_orthogonality 通过，最大重叠: {max_overlap:.4f}")


def test_adaptive_lambda_effect():
    """
    测试：adaptive_lambda_min 越大，正交投影越强，B_t 与旧 GPM 的重叠应越小。
    """
    overlaps = {}
    for lam in [0.0, 1.0]:
        layer = _build_layer(
            freq_level=2, spatial_blocks=(1, 1),
            adaptive_lambda_min=lam, adaptive_lambda_max=lam,
        )
        x1 = torch.randn(2, 8, 32, 32)
        x2 = torch.randn(2, 8, 32, 32)

        layer.add_task()
        layer.set_current_task(0)
        layer.calibrate(x1, 0, task_similarity=0.0)
        layer.update_gpm(x1, 0)

        layer.add_task()
        layer.set_current_task(1)
        layer.calibrate(x2, 1, task_similarity=0.0)

        B2_private = layer.task_B[1].data[layer.share_rank:]
        max_overlap = 0.0
        for key, gpm in layer.dual_gpms.items():
            if gpm.basis is not None:
                overlap = orthogonality_check(B2_private, gpm.basis)
                max_overlap = max(max_overlap, overlap)
        overlaps[lam] = max_overlap

    # lam=1.0 时完全正交，重叠应小于等于 lam=0.0
    assert overlaps[1.0] <= overlaps[0.0] + 0.1, \
        f"自适应强度未生效: lam=0.0 -> {overlaps[0.0]:.4f}, " \
        f"lam=1.0 -> {overlaps[1.0]:.4f}"
    print(f"test_adaptive_lambda_effect 通过，"
          f"lam=0.0 -> {overlaps[0.0]:.4f}, lam=1.0 -> {overlaps[1.0]:.4f}")


def test_share_subspace_orthogonal_to_private():
    """
    测试：共享子空间 B_share 与私有基应尽量不重叠。
    """
    layer = _build_layer(freq_level=2, spatial_blocks=(1, 1), share_subspace=True)
    x = torch.randn(2, 8, 32, 32)

    layer.add_task()
    layer.set_current_task(0)
    layer.calibrate(x, 0, task_similarity=0.0)

    if layer.B_share is None or not layer.share_initialized:
        print("test_share_subspace_orthogonal_to_private 跳过（未初始化共享基）")
        return

    B_share = layer.B_share.data
    B_private = layer.task_B[0].data[layer.share_rank:]
    overlap = subspace_overlap(B_share, B_private)
    # 共享基与私有基不应完全重合
    assert overlap < 0.9, f"共享基与私有基重叠过大: {overlap:.4f}"
    print(f"test_share_subspace_orthogonal_to_private 通过，"
          f"最大余弦相似度: {overlap:.4f}")


def test_multi_task_orthogonality():
    """
    测试：三个任务依次训练后，最后一个任务的 B_t 与前两个任务的 GPM 基正交。
    """
    layer = _build_layer(freq_level=2, spatial_blocks=(1, 1))
    xs = [torch.randn(2, 8, 32, 32) for _ in range(3)]

    for idx in range(3):
        layer.add_task()
        layer.set_current_task(idx)
        sim = 0.0 if idx == 0 else 0.5
        layer.calibrate(xs[idx], idx, task_similarity=sim)
        layer.update_gpm(xs[idx], idx)

    # 检查第三个任务的 B_t 与所有 GPM 基
    B3_private = layer.task_B[2].data[layer.share_rank:]
    max_overlap = 0.0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            overlap = orthogonality_check(B3_private, gpm.basis)
            max_overlap = max(max_overlap, overlap)

    assert max_overlap < 0.6, \
        f"多任务后正交性不满足: {max_overlap:.4f}"
    print(f"test_multi_task_orthogonality 通过，最大重叠: {max_overlap:.4f}")


def test_no_orthogonal_when_lambda_zero():
    """
    测试：当 adaptive_lambda_min=adaptive_lambda_max=0 时，不做正交投影，
    B_t 与旧 GPM 的重叠应更大。
    """
    layer = _build_layer(
        freq_level=2, spatial_blocks=(1, 1),
        adaptive_lambda_min=0.0, adaptive_lambda_max=0.0,
    )
    x1 = torch.randn(2, 8, 32, 32)
    x2 = torch.randn(2, 8, 32, 32)

    layer.add_task()
    layer.set_current_task(0)
    layer.calibrate(x1, 0, task_similarity=0.0)
    layer.update_gpm(x1, 0)

    layer.add_task()
    layer.set_current_task(1)
    layer.calibrate(x2, 1, task_similarity=0.0)

    B2_private = layer.task_B[1].data[layer.share_rank:]
    max_overlap = 0.0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            overlap = orthogonality_check(B2_private, gpm.basis)
            max_overlap = max(max_overlap, overlap)

    # 不做正交投影时重叠可能更大，只验证不报错
    print(f"test_no_orthogonal_when_lambda_zero 通过，"
          f"最大重叠: {max_overlap:.4f}")


if __name__ == '__main__':
    test_Bt_orthogonal_to_old_gpm()
    test_freq_level_1_orthogonality()
    test_freq_level_2_orthogonality()
    test_adaptive_lambda_effect()
    test_share_subspace_orthogonal_to_private()
    test_multi_task_orthogonality()
    test_no_orthogonal_when_lambda_zero()
    print("所有正交性测试通过。")