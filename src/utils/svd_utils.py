import torch
import math


# ============================================================
# 工具函数
# ============================================================

def _sanitize(matrix):
    """清理 NaN / Inf，避免 SVD 失败。"""
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _clamp_k(k, m, n):
    """把 k 限制在 [1, min(m, n)] 之间。"""
    if k is None or k <= 0:
        return 1
    return max(1, min(int(k), min(m, n)))


# ============================================================
# 随机 SVD
# ============================================================

def random_svd_topk(matrix, k, oversample=10, niter=2):
    """
    随机 SVD，返回 top-k 右奇异向量。
    matrix: [m, n]
    返回: V [k, n]

    步骤：
      1. 生成随机投影 Omega: [n, k+p]
      2. Y = matrix @ Omega
      3. QR 分解
      4. 幂迭代精化
      5. 在低维空间做 SVD，取右奇异向量
    """
    matrix = _sanitize(matrix)
    m, n = matrix.shape
    k = _clamp_k(k, m, n)

    # 过采样参数
    p = min(oversample, max(0, min(m, n) - k))
    q = min(k + p, min(m, n))
    if q <= 0:
        return torch.zeros(k, n, device=matrix.device, dtype=matrix.dtype)

    # 随机投影
    Omega = torch.randn(n, q, device=matrix.device, dtype=matrix.dtype)
    Y = matrix @ Omega

    # QR 分解
    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # 幂迭代
    for _ in range(niter):
        Y2 = matrix.T @ Q
        Q2, _ = torch.linalg.qr(Y2, mode='reduced')
        Y = matrix @ Q2
        Q, _ = torch.linalg.qr(Y, mode='reduced')

    # 在低维空间做 SVD
    B = Q.T @ matrix  # [q, n]
    try:
        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    except Exception:
        # 回退到完整 SVD
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # 取前 k 个右奇异向量
    V = Vh[:k, :]
    return V


# ============================================================
# 截断 SVD
# ============================================================

def truncated_svd_topk(matrix, k):
    """
    截断 SVD，返回 top-k 右奇异向量。
    优先使用 torch.svd_lowrank，若失败则回退到完整 SVD。
    """
    matrix = _sanitize(matrix)
    m, n = matrix.shape
    k = _clamp_k(k, m, n)

    if k >= min(m, n):
        # 直接完整 SVD
        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            return Vh[:k, :]
        except Exception:
            return torch.zeros(k, n, device=matrix.device, dtype=matrix.dtype)

    try:
        U, S, Vh = torch.svd_lowrank(matrix, q=k, niter=2)
        return Vh[:k, :]
    except Exception:
        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            return Vh[:k, :]
        except Exception:
            return torch.zeros(k, n, device=matrix.device, dtype=matrix.dtype)


# ============================================================
# 统一入口
# ============================================================

def topk_right_singular(matrix, k, method='random', oversample=10, niter=2):
    """
    返回 top-k 右奇异向量。
    method: 'random' | 'truncated' | 'full'
    """
    if method == 'random':
        return random_svd_topk(matrix, k, oversample=oversample, niter=niter)
    elif method == 'truncated':
        return truncated_svd_topk(matrix, k)
    elif method == 'full':
        matrix = _sanitize(matrix)
        m, n = matrix.shape
        k = _clamp_k(k, m, n)
        try:
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            return Vh[:k, :]
        except Exception:
            return torch.zeros(k, n, device=matrix.device, dtype=matrix.dtype)
    else:
        raise ValueError(f"未知 SVD 方法: {method}")


# ============================================================
# 自适应 rank：基于累计能量
# ============================================================

def energy_based_rank(matrix, energy_ratio=0.99, max_rank=None):
    """
    根据累计能量自动选择 rank。
    matrix: [m, n]
    返回: int rank
    """
    matrix = _sanitize(matrix)
    m, n = matrix.shape
    max_r = min(m, n) if max_rank is None else min(max_rank, min(m, n))
    if max_r <= 0:
        return 1
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return 1
    energy = S ** 2
    total = energy.sum()
    if total <= 1e-12:
        return 1
    cumsum = torch.cumsum(energy, dim=0)
    k = int(torch.searchsorted(cumsum, energy_ratio * total).item()) + 1
    return max(1, min(k, max_r))


# ============================================================
# 正交性检查工具
# ============================================================

def orthogonality_check(A, B):
    """
    检查 A 的行空间与 B 的列空间的重叠程度。
    A: [r, d]
    B: [d, k]
    返回: float，越小越正交
    """
    if A is None or B is None or A.numel() == 0 or B.numel() == 0:
        return 0.0
    A = _sanitize(A)
    B = _sanitize(B)
    proj = A @ B  # [r, k]
    # 归一化：除以 A 和 B 的 Frobenius 范数
    denom = (A.norm() * B.norm() + 1e-12)
    return (proj.norm() / denom).item()


def subspace_overlap(A, B):
    """
    计算两个子空间的余弦相似度矩阵的最大值。
    A: [r1, d]
    B: [r2, d]
    返回: float
    """
    if A is None or B is None or A.numel() == 0 or B.numel() == 0:
        return 0.0
    A = _sanitize(A)
    B = _sanitize(B)
    A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    sim = A_norm @ B_norm.T  # [r1, r2]
    return sim.abs().max().item()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "random_svd_topk",
    "truncated_svd_topk",
    "topk_right_singular",
    "energy_based_rank",
    "orthogonality_check",
    "subspace_overlap",
]