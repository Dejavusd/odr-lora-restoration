import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dual_gpm import DualGPM


def test_update_project():
    """基础测试：更新后投影到正交补，结果与旧基正交。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H1 = torch.randn(16, 100)
    gpm.update(H1)
    assert gpm.basis is not None, "basis 未更新"

    H2 = torch.randn(16, 50)
    H2_hat = gpm.project_out(H2)

    # 投影后与旧基正交
    proj = gpm.basis.T @ H2_hat
    assert proj.abs().max() < 1e-4, f"投影后未正交: {proj.abs().max().item()}"
    print("test_update_project 通过")


def test_project_in():
    """测试投影到旧基张成的空间。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H1 = torch.randn(16, 100)
    gpm.update(H1)

    H2 = torch.randn(16, 50)
    H2_in = gpm.project_in(H2)

    # 投影结果应完全落在旧基张成的子空间内
    # 检查 H2_in 是否满足 M M^T H2_in = H2_in
    proj = gpm.basis @ (gpm.basis.T @ H2_in)
    assert torch.allclose(H2_in, proj, atol=1e-4), \
        f"project_in 结果不在旧基张成的子空间内"
    print("test_project_in 通过")


def test_empty_basis():
    """测试 basis 为空时的行为。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H = torch.randn(16, 50)

    # 未更新前，project_out 应返回原输入
    H_out = gpm.project_out(H)
    assert torch.allclose(H_out, H), "空 basis 时 project_out 应返回原输入"

    # 未更新前，project_in 应返回零
    H_in = gpm.project_in(H)
    assert torch.allclose(H_in, torch.zeros_like(H)), \
        "空 basis 时 project_in 应返回零"
    print("test_empty_basis 通过")


def test_zero_input():
    """测试零矩阵输入不会破坏 DualGPM。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H = torch.zeros(16, 100)
    gpm.update(H)
    # 零矩阵不应产生有效基
    assert gpm.basis is None, "零矩阵不应产生有效基"
    print("test_zero_input 通过")


def test_nan_input():
    """测试 NaN 输入不会导致崩溃。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H = torch.randn(16, 100)
    H[0, 0] = float('nan')
    gpm.update(H)
    # 应能正常执行，不抛异常
    print("test_nan_input 通过")


def test_multi_gpm_dict():
    """测试多个 DualGPM 按 (g, f) 管理，模拟多空间块 + 多子带场景。"""
    grid_h, grid_w = 2, 2
    freq_bands = 7
    dim = 32
    gpms = {}
    for g in range(grid_h * grid_w):
        for f in range(freq_bands):
            gpms[(g, f)] = DualGPM(dim=dim, eps=0.99)

    # 更新所有 GPM
    for g in range(grid_h * grid_w):
        for f in range(freq_bands):
            H = torch.randn(dim, 100)
            gpms[(g, f)].update(H)

    # 检查所有 GPM 都有基
    for key, gpm in gpms.items():
        assert gpm.basis is not None, f"GPM {key} 未更新"

    # 投影并检查正交性
    for key, gpm in gpms.items():
        H2 = torch.randn(dim, 50)
        H2_hat = gpm.project_out(H2)
        proj = gpm.basis.T @ H2_hat
        assert proj.abs().max() < 1e-4, f"GPM {key} 投影后未正交"
    print("test_multi_gpm_dict 通过")


def test_state_dict_save_load():
    """测试 state_dict 保存与加载。"""
    gpm = DualGPM(dim=16, eps=0.99)
    H = torch.randn(16, 100)
    gpm.update(H)
    state = gpm.state_dict()

    # 新建 GPM 并加载
    gpm2 = DualGPM(dim=16, eps=0.99)
    assert gpm2.basis is None
    gpm2.load_state_dict(state)
    assert gpm2.basis is not None
    assert torch.allclose(gpm.basis, gpm2.basis, atol=1e-6), \
        "state_dict 加载后 basis 不一致"
    print("test_state_dict_save_load 通过")


def test_incremental_update():
    """测试多次 update 后的基合并与正交性保持。"""
    gpm = DualGPM(dim=16, eps=0.99)

    # 第一次更新
    H1 = torch.randn(16, 100)
    gpm.update(H1)
    k1 = gpm.basis.shape[1]

    # 第二次更新，维度应与旧任务正交的部分为主
    H2 = torch.randn(16, 100)
    gpm.update(H2)
    k2 = gpm.basis.shape[1]

    # 基数量应增长或保持不变，不超过 dim
    assert k2 >= 1 and k2 <= 16, f"basis 维度异常: {k2}"

    # 投影后与最终基正交
    H3 = torch.randn(16, 50)
    H3_hat = gpm.project_out(H3)
    proj = gpm.basis.T @ H3_hat
    assert proj.abs().max() < 1e-4, f"增量更新后投影未正交"
    print(f"test_incremental_update 通过，basis 从 {k1} 变为 {k2}")


def test_max_basis():
    """测试 max_basis 限制生效。"""
    gpm = DualGPM(dim=16, eps=0.99, max_basis=4)
    for _ in range(5):
        H = torch.randn(16, 100)
        gpm.update(H)
    assert gpm.basis.shape[1] <= 4, \
        f"basis 维度超过 max_basis: {gpm.basis.shape[1]}"
    print(f"test_max_basis 通过，basis 维度: {gpm.basis.shape[1]}")


def test_eps_effect():
    """测试 eps 对 basis 维度的影响。"""
    H = torch.randn(16, 100)
    gpm_low = DualGPM(dim=16, eps=0.5)
    gpm_high = DualGPM(dim=16, eps=0.99)
    gpm_low.update(H)
    gpm_high.update(H)

    # eps 越大，保留的基越多或相等
    assert gpm_high.basis.shape[1] >= gpm_low.basis.shape[1], \
        f"eps 更大时 basis 维度应更大: " \
        f"{gpm_high.basis.shape[1]} vs {gpm_low.basis.shape[1]}"
    print(f"test_eps_effect 通过，eps=0.5: {gpm_low.basis.shape[1]}, "
          f"eps=0.99: {gpm_high.basis.shape[1]}")


if __name__ == '__main__':
    test_update_project()
    test_project_in()
    test_empty_basis()
    test_zero_input()
    test_nan_input()
    test_multi_gpm_dict()
    test_state_dict_save_load()
    test_incremental_update()
    test_max_basis()
    test_eps_effect()
    print("所有测试通过。")