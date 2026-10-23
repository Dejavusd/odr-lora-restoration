import torch
import torch.nn as nn


class DualGPM(nn.Module):
    """
    Dual Gradient Projection Memory.
    维护旧任务梯度空间 M 或正交补 M_perp，无需旧数据。
    每层维护一个有界的激活子空间。

    参数：
        dim: 输入特征维度 d = C_in * K * K
        eps: 累计能量阈值
        max_basis: 最大基向量数，None 表示不限制
        use_orthogonal_complement: 是否维护正交补（本实现统一维护 M）
    """

    def __init__(self, dim, eps=0.99, max_basis=None, use_orthogonal_complement=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.max_basis = max_basis if max_basis is not None else dim
        self.use_orthogonal_complement = use_orthogonal_complement
        # Register buffers so GPM statistics are part of model checkpoints.
        # Empty tensors represent an uninitialized basis while keeping the
        # public ``basis is None`` API used throughout the project.
        self.register_buffer('_basis', torch.empty(dim, 0))
        self.register_buffer('_singular_values', torch.empty(0))

    @property
    def basis(self):
        return None if self._basis.numel() == 0 else self._basis

    @basis.setter
    def basis(self, value):
        if value is None:
            self._basis = torch.empty(
                self.dim, 0, device=self._basis.device, dtype=self._basis.dtype
            )
        else:
            self._basis = value

    @property
    def singular_values(self):
        return None if self._singular_values.numel() == 0 else self._singular_values

    @singular_values.setter
    def singular_values(self, value):
        if value is None:
            self._singular_values = torch.empty(
                0, device=self._singular_values.device,
                dtype=self._singular_values.dtype,
            )
        else:
            self._singular_values = value

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # GPM rank is data-dependent, so resize buffers before PyTorch checks
        # checkpoint tensor shapes.
        basis_key = prefix + '_basis'
        singular_key = prefix + '_singular_values'
        if basis_key in state_dict:
            self._basis = torch.empty_like(state_dict[basis_key])
        if singular_key in state_dict:
            self._singular_values = torch.empty_like(state_dict[singular_key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def update(self, H):
        """
        用当前任务的输入矩阵 H 更新旧任务梯度空间。
        H: [dim, n]
        """
        if H is None or H.numel() == 0 or H.shape[1] == 0:
            return
        H = torch.nan_to_num(
            H.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
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

