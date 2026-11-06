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

    @torch.no_grad()
    def extend_from_covariance(self, covariance, max_basis=None):
        """Return a larger basis without dropping any protected directions.

        Covariance is an uncentered sum H H^T in the actual convolution input
        coordinates. Fit residual PCA in float64 on CPU; do not change self.
        max_basis is a cap, not a required rank. eps controls sampled energy.
        """
        cap = self.max_basis if max_basis is None else int(max_basis)
        cap = min(cap, self.dim)
        if cap < 1 or covariance.shape != (self.dim, self.dim):
            raise ValueError('Invalid covariance shape or GPM capacity')
        C = covariance.detach().to(device='cpu', dtype=torch.float64)
        C = (C + C.T) * 0.5
        if not torch.isfinite(C).all():
            raise ValueError('Non-finite activation covariance')
        total = float(C.trace())
        old = self.basis
        M = (torch.empty(self.dim, 0, dtype=torch.float64) if old is None
             else torch.linalg.qr(old.detach().cpu().double(), mode='reduced').Q)
        old_rank = M.shape[1]
        if cap < old_rank:
            raise ValueError('Cannot shrink capacity without dropping old directions')
        protected = float((M.T @ C @ M).trace()) if old_rank else 0.0
        missing = max(0.0, self.eps * total - protected)
        k = 0
        if missing > max(1e-12, total * 1e-12) and cap > old_rank:
            residual = C - M @ (M.T @ C)
            residual = residual - (residual @ M) @ M.T
            residual = (residual + residual.T) * 0.5
            eigenvalues, eigenvectors = torch.linalg.eigh(residual)
            energy = eigenvalues.flip(0).clamp_min(0)
            vectors = eigenvectors.flip(1)
            valid = int((energy > max(1e-12, float(energy[0]) * 1e-7)).sum())
            requested = int(torch.searchsorted(energy.cumsum(0), missing)) + 1
            k = min(requested, valid, cap - old_rank, self.dim - old_rank)
            if k:
                V = vectors[:, :k]
                for _ in range(2):
                    V = V - M @ (M.T @ V)
                V = torch.linalg.qr(V, mode='reduced').Q
                M = torch.cat((M, V), dim=1)
        achieved = float((M.T @ C @ M).trace()) / max(total, 1e-12)
        ref = self._basis
        return M.to(device=ref.device, dtype=ref.dtype), dict(
            dim=self.dim, cap=cap, old_rank=old_rank, added_rank=k,
            rank=M.shape[1], sampled_coverage_before=protected / max(total, 1e-12),
            sampled_coverage_after=achieved, total_energy=total)

    @torch.no_grad()
    def accumulate_balanced(self, H):
        if not torch.isfinite(H).all():
            raise FloatingPointError('Nonfinite balanced GPM statistics')
        contribution=H.detach().float()@H.detach().float().T
        if getattr(self,'_balanced_covariance',None) is None:
            self._balanced_covariance=contribution
            self._balanced_columns=H.shape[1]
        else:
            self._balanced_covariance.add_(contribution)
            self._balanced_columns+=H.shape[1]

    @torch.no_grad()
    def finalize_balanced(self, old_fraction=.5):
        """One aggregate covariance per task; reserve old-space quota then residual PCA.

        No historical eigenvalue ordering is available. Retain old-space modes
        least active on current inputs, leaving current PCA to cover overlap.
        This is a stated allocation heuristic, not old-task importance ranking.
        """
        import math
        if not math.isfinite(old_fraction) or not 0<=old_fraction<=1:
            raise ValueError('old_fraction must be in [0,1]')
        covariance=getattr(self,'_balanced_covariance',None)
        if covariance is None:raise ValueError('No balanced covariance collected')
        C=covariance.cpu().double();C=(C+C.T)*.5
        if not torch.isfinite(C).all():raise FloatingPointError('Nonfinite covariance')
        import hashlib
        covariance_sha256=hashlib.sha256(C.contiguous().numpy().tobytes()).hexdigest()
        total=float(C.trace());cap=min(self.max_basis,self.dim)
        old=(torch.empty(self.dim,0,dtype=torch.float64) if self.basis is None
             else torch.linalg.qr(self.basis.cpu().double(),mode='reduced').Q)
        old_rank=old.shape[1]
        if old_rank>cap:raise ValueError('Existing basis exceeds capacity')
        budget=min(old_rank,int(math.floor(cap*old_fraction+.5)))
        if total<=1e-20:
            result=old;kept=old_rank;added=0
        else:
            C=C/total
            if budget==old_rank:
                retained=old
            elif budget:
                _,vectors=torch.linalg.eigh(old.T@C@old)
                retained=old@vectors[:,:budget]
            else:
                retained=old[:,:0]
            kept=retained.shape[1]
            residual=C-retained@(retained.T@C)
            residual=residual-(residual@retained)@retained.T
            residual=(residual+residual.T)*.5
            protected=float((retained.T@C@retained).trace()) if kept else 0.
            missing=max(0.,self.eps-protected)
            room=cap-kept;added=0;result=retained
            if room and missing>1e-12:
                eigenvalues,vectors=torch.linalg.eigh(residual)
                energy=eigenvalues.flip(0).clamp_min(0);vectors=vectors.flip(1)
                valid=int((energy>max(1e-12,float(energy[0])*1e-7)).sum())
                requested=int(torch.searchsorted(energy.cumsum(0),missing))+1
                added=min(room,valid,requested)
                if added:
                    new=vectors[:,:added]
                    for _ in range(2):new=new-retained@(retained.T@new)
                    new=torch.linalg.qr(new,mode='reduced').Q
                    result=torch.cat((retained,new),1)
        if result.shape[1]==0:raise ValueError('Empty balanced memory')
        if float((result.T@result-torch.eye(result.shape[1],dtype=result.dtype)).abs().max())>1e-5:
            raise RuntimeError('Balanced basis not orthonormal')
        residual=float((old-result@(result.T@old)).norm()/old.norm().clamp_min(1e-20)) if old_rank else 0.
        info=dict(cap=cap,old_fraction=old_fraction,old_rank=old_rank,old_quota=budget,
            retained_old_rank=kept,current_residual_rank=added,rank=result.shape[1],
            columns=self._balanced_columns,old_span_residual=residual,total_energy=total,
            old_span_overlap_fraction=float((old.T@result).square().sum()/old_rank) if old_rank else 0.,
            current_energy_coverage=float((result.T@C@result).trace()) if total>1e-20 else 0.,
            covariance_sha256=covariance_sha256,
            selector='least current-energy modes inside old span, then current residual PCA',
            no_current_energy=total<=1e-20)
        self.basis=result.to(device=self._basis.device,dtype=self._basis.dtype)
        self.singular_values=None
        self._balanced_covariance=None;self._balanced_columns=0
        return info

    @torch.no_grad()
    def update_preserving(self, H):
        if H is None or not H.numel():
            return
        H = H.detach().float()
        basis, info = self.extend_from_covariance(H @ H.T)
        self.basis = basis
        self.singular_values = None
        return info

    def project_in(self, H):
        """投影到 M 内。"""
        if self.basis is None:
            return torch.zeros_like(H)
        M = self.basis
        return M @ (M.T @ H)

