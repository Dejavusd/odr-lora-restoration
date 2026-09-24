import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.odr_lora import ODRLoRAConv2d
from src.models.spatial_block import num_subbands


def test_forward_shapes():
    """测试单任务前向输出形状。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    y = layer(x, 0)
    assert y.shape == (2, 16, 32, 32), y.shape
    print("test_forward_shapes 通过")


def test_forward_no_task_id():
    """测试不传 task_idx 时使用 current_task_idx。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    y = layer(x)
    assert y.shape == (2, 16, 32, 32), y.shape
    print("test_forward_no_task_id 通过")


def test_subband_count():
    """测试 freq_level=1 和 freq_level=2 的子带数量。"""
    assert num_subbands(level=1) == 4
    assert num_subbands(level=2) == 7

    layer1 = ODRLoRAConv2d(8, 16, kernel_size=3, freq_level=1)
    layer2 = ODRLoRAConv2d(8, 16, kernel_size=3, freq_level=2)

    grid_h, grid_w = layer1.spatial_blocks
    assert len(layer1.dual_gpms) == grid_h * grid_w * 4
    grid_h, grid_w = layer2.spatial_blocks
    assert len(layer2.dual_gpms) == grid_h * grid_w * 7
    print("test_subband_count 通过")


def test_calibrate():
    """测试 calibrate 后 B_t 非零，且共享基被初始化。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3, spatial_blocks=(2, 2), freq_level=2)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    layer.calibrate(x, 0)

    B = layer.task_B[0]
    assert B.abs().sum() > 0, "B_t 未更新"

    if layer.B_share is not None:
        assert layer.share_initialized, "共享基未初始化"
        # 前 share_rank 行应等于 B_share
        share_part = B.data[:layer.share_rank]
        assert torch.allclose(share_part, layer.B_share.data, atol=1e-5), \
            "B_t 的共享部分与 B_share 不一致"
        # 私有部分应非零
        private_part = B.data[layer.share_rank:]
        assert private_part.abs().sum() > 0, "私有基未更新"

    print("test_calibrate 通过")


def test_calibrate_freq_level_1():
    """测试 freq_level=1 时 calibrate 仍可运行。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3, spatial_blocks=(2, 2), freq_level=1)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    layer.calibrate(x, 0)
    B = layer.task_B[0]
    assert B.abs().sum() > 0
    print("test_calibrate_freq_level_1 通过")


def test_add_multiple_tasks():
    """测试多任务分支添加与切换。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3)
    layer.add_task()
    layer.add_task()
    layer.add_task()
    assert len(layer.task_A) == 3
    assert len(layer.task_B) == 3

    layer.set_current_task(1)
    assert layer.current_task_idx == 1

    x = torch.randn(2, 8, 32, 32)
    for idx in range(3):
        y = layer(x, idx)
        assert y.shape == (2, 16, 32, 32)
    print("test_add_multiple_tasks 通过")


def test_freeze_base_and_old():
    """测试冻结基座和旧分支后，只有当前任务 A 可训练。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3)
    layer.add_task()
    layer.add_task()
    layer.set_current_task(1)
    layer.freeze_base_and_old(1)

    # 基座不可训练
    assert not layer.weight.requires_grad
    if layer.bias is not None:
        assert not layer.bias.requires_grad

    # 旧任务 A/B 都不可训练
    assert not layer.task_A[0].requires_grad
    assert not layer.task_B[0].requires_grad

    # 当前任务 A 可训练，B 不可训练
    assert layer.task_A[1].requires_grad
    assert not layer.task_B[1].requires_grad

    print("test_freeze_base_and_old 通过")


def test_merge_task():
    """测试 merge_task 后基座权重发生变化。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    layer.calibrate(x, 0)

    # 给 A 赋值，使其不为零
    with torch.no_grad():
        layer.task_A[0].normal_(0, 0.1)

    w_before = layer.weight.data.clone()
    layer.merge_task(0)
    w_after = layer.weight.data.clone()

    assert not torch.allclose(w_before, w_after), "merge_task 未改变基座权重"
    print("test_merge_task 通过")


def test_update_gpm():
    """测试 update_gpm 后 DualGPM 基非空。"""
    layer = ODRLoRAConv2d(8, 16, kernel_size=3, spatial_blocks=(1, 1), freq_level=2)
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 8, 32, 32)
    layer.update_gpm(x, 0)

    non_empty = 0
    for key, gpm in layer.dual_gpms.items():
        if gpm.basis is not None:
            non_empty += 1
    assert non_empty > 0, "所有 DualGPM 都为空"
    print(f"test_update_gpm 通过，非空 GPM 数量: {non_empty}")


if __name__ == '__main__':
    test_forward_shapes()
    test_forward_no_task_id()
    test_subband_count()
    test_calibrate()
    test_calibrate_freq_level_1()
    test_add_multiple_tasks()
    test_freeze_base_and_old()
    test_merge_task()
    test_update_gpm()
    print("所有测试通过。")