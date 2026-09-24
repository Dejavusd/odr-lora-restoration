import torch
import torch.nn as nn


class DualGPM:
    """
    Dual Gradient Projection Memory.
    维护旧任务梯度空间 M 或正交补 M_perp，无需旧数据。
    针对每层、每空间块、每频带独立维护。

    参数：
        dim: 输入特征维度 d = C_in * K * K
        eps: 累计能量阈值
        max_basis: 最大基向量数，None 表示不限制
        use_orthogonal_complement: 是否维护正交补（本实现统一维护 M）
    """

    def __init__(self, dim, eps=0.99, max_basis=None, use_orthogonal_complement=False):
        self.dim = dim
        self.eps = eps
        self.max_basis = max_basis if max_basis is not None else dim
        self.use_orthogonal_complement = use_orthogonal_complement
        self.basis = None          # [dim, k]
        self.singular_values = None

    def update(self, H):
        """
        用当前任务的输入矩阵 H 更新旧任务梯度空间。
        H: [dim, n]
        """
        if H is None or H.numel() == 0 or H.shape[1] == 0:
            return
        H = H.float()
        # SVD 分解
        try:
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        except Exception:
            return
        energy = S ** 2
        total = energy.sum()
        if total <= 1e-12:
            return
        cumsum = torch.cumsum(energy, dim=0)
        k = int(torch.searchsorted(cumsum, self.eps * total).item()) + 1
        k = min(k, self.max_basis, U.shape[1])
        new_basis = U[:, :k]  # [dim, k]

        if self.basis is None:
            self.basis = new_basis
        else:
            combined = torch.cat([self.basis, new_basis], dim=1)
            U2, S2, _ = torch.linalg.svd(combined, full_matrices=False)
            energy2 = S2 ** 2
            total2 = energy2.sum()
            cumsum2 = torch.cumsum(energy2, dim=0)
            k2 = int(torch.searchsorted(cumsum2, self.eps * total2).item()) + 1
            k2 = min(k2, self.max_basis, U2.shape[1])
            self.basis = U2[:, :k2]
        self.singular_values = S

    def project_out(self, H):
        """
        将 H 投影到旧任务梯度空间的正交补。
        H: [dim, n]
        返回: H_hat [dim, n]
        """
        if self.basis is None:
            return H
        M = self.basis  # [dim, k]
        proj = M @ (M.T @ H)
        return H - proj

    def project_in(self, H):
        """投影到 M 内。"""
        if self.basis is None:
            return torch.zeros_like(H)
        M = self.basis
        return M @ (M.T @ H)

    def state_dict(self):
        return {
            'basis': self.basis.cpu() if self.basis is not None else None,
            'singular_values': self.singular_values.cpu() if self.singular_values is not None else None,
            'dim': self.dim,
            'eps': self.eps,
            'max_basis': self.max_basis,
        }

    def load_state_dict(self, state):
        self.basis = state['basis'].to(torch.float32) if state['basis'] is not None else None
        self.singular_values = state['singular_values'] if state['singular_values'] is not None else None