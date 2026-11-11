"""Frozen per-task activation sketches and fixed-capacity shared protection.

No images are retained. Sketch coverage is NOT coverage of future activations.
The balanced selector is a greedy heuristic, not a global max-min optimizer.
"""
import hashlib
import math

import torch


class TaskSketchAccumulator:
    """Streaming truncated SVD of current-task patches, bounded by dim x rank.

    A factor F approximates sum(H H^T) as F F^T. Repeated truncation is lossy
    and order-dependent. Full sampled energy is tracked separately for audit.
    """
    def __init__(self, dim, rank=32):
        if min(dim, rank) < 1:
            raise ValueError('Positive sketch dimension and rank required')
        self.dim, self.rank = dim, min(dim, rank)
        self.factor = None
        self.columns = 0
        self.total_energy = 0.

    @torch.no_grad()
    def add(self, H):
        if H.ndim != 2 or H.shape[0] != self.dim or not H.shape[1]:
            raise ValueError('Invalid task sketch input')
        H = H.detach().float()
        if not torch.isfinite(H).all():
            raise FloatingPointError('Non-finite task sketch input')
        energy = float(H.square().sum(dtype=torch.float64))
        joined = H if self.factor is None else torch.cat((self.factor, H), 1)
        U, S, _ = torch.linalg.svd(joined, full_matrices=False)
        threshold = max(1e-12, float(S[0]) * 1e-7)
        count = min(self.rank, int((S > threshold).sum()))
        self.factor = (U[:, :count] * S[:count]).contiguous()
        self.columns += H.shape[1]
        self.total_energy += energy

    def finish(self):
        if self.factor is None or not self.columns:
            raise ValueError('No current-task statistics collected')
        return dict(factor=self.factor.detach().cpu().float().clone(), columns=self.columns,
                    total_energy=self.total_energy)


def sketch_hash(sketch):
    h = hashlib.sha256()
    factor = sketch['factor'].detach().cpu().contiguous()
    h.update(str((tuple(factor.shape), str(factor.dtype), sketch['columns'], sketch['total_energy'])).encode())
    h.update(factor.numpy().tobytes())
    return h.hexdigest()


def bank_hash(bank):
    h = hashlib.sha256()
    for name in sorted(bank):
        for task in sorted(bank[name], key=int):
            h.update(name.encode()); h.update(task.encode())
            h.update(sketch_hash(bank[name][task]).encode())
    return h.hexdigest()


@torch.no_grad()
def select_task_memory(sketches, cap, mode='balanced'):
    """One trace-normalized covariance per task; joint PCA or worst-first greedy.

    Work in the union span (at most tasks * sketch_rank), never materialize
    a dim x dim covariance. Fill the same feasible rank in both modes.
    """
    if mode not in ('balanced', 'mean_pca') or cap < 1 or not sketches:
        raise ValueError('Invalid task memory mode/capacity/bank')
    tasks = sorted(sketches, key=int)
    dim = None
    normalized, active = [], []
    entries = {}
    for task in tasks:
        entry = sketches[task]
        F = entry['factor'].detach().cpu().double()
        if F.ndim != 2 or F.shape[0] < 1 or (dim is not None and F.shape[0] != dim):
            raise ValueError('Incompatible task sketch dimensions')
        dim = F.shape[0]
        total = float(entry['total_energy'])
        if not torch.isfinite(F).all() or not math.isfinite(total) or total < 0 or entry['columns'] < 1:
            raise ValueError('Invalid task sketch values')
        retained = float(F.square().sum())
        entries[task] = dict(columns=entry['columns'], sketch_rank=F.shape[1],
            retained_energy_fraction=retained/max(total, 1e-30), full_sampled_energy=total,
            retained_sketch_energy=retained, sketch_sha256=sketch_hash(entry),
            sketch_bytes=entry['factor'].numel()*entry['factor'].element_size())
        if retained > 1e-20:
            normalized.append(F / math.sqrt(retained)); active.append(task)
    if not normalized:
        raise ValueError('All task sketches have zero energy')
    U, S, _ = torch.linalg.svd(torch.cat(normalized, 1), full_matrices=False)
    union_rank = int((S > max(1e-12, float(S[0])*1e-9)).sum())
    Q = U[:, :union_rank]
    target_rank = min(cap, union_rank, dim)
    reduced = [Q.T @ F for F in normalized]
    covariances = [(G @ G.T) for G in reduced]
    pooled = sum(covariances)/len(covariances)
    choices = []
    if mode == 'mean_pca' or len(normalized) == 1:
        # Identical construction for the first task gives identical RainH starts.
        _, vectors = torch.linalg.eigh(pooled)
        V = vectors.flip(1)[:, :target_rank]
    else:
        V = torch.empty(union_rank, 0, dtype=torch.float64)
        identity = torch.eye(union_rank, dtype=torch.float64)
        for _ in range(target_rank):
            scores = [float((V.T @ C @ V).trace()) for C in covariances]
            worst = min(range(len(scores)), key=lambda i: (scores[i], int(active[i])))
            P = identity - V @ V.T
            residual = P @ covariances[worst] @ P
            residual = (residual + residual.T)*.5
            energy, vectors = torch.linalg.eigh(residual)
            fallback = float(energy[-1]) <= 1e-12
            if fallback:
                residual = P @ pooled @ P
                _, vectors = torch.linalg.eigh((residual+residual.T)*.5)
            direction = vectors[:, -1:]
            for _ in range(2):
                direction = direction - V @ (V.T @ direction)
            length = float(direction.norm())
            if length < 1e-10:
                raise RuntimeError('Unable to fill feasible task memory rank')
            V = torch.cat((V, direction/length), 1)
            choices.append(dict(task=active[worst], coverage_before=scores[worst], fallback=fallback))
    basis = Q @ V
    basis = torch.linalg.qr(basis, mode='reduced').Q
    # Canonical signs improve reproducibility without changing the projection.
    pivots = basis.abs().argmax(0)
    signs = basis[pivots, torch.arange(basis.shape[1])].sign()
    basis = basis * signs
    for task, F in zip(active, normalized):
        entries[task]['coverage'] = min(1., max(0., float((basis.T @ F).square().sum())))
    for task in tasks:
        if task not in active:
            entries[task]['coverage'] = None
    scores = [entries[t]['coverage'] for t in active]
    return basis.float(), dict(mode=mode, cap=cap, rank=basis.shape[1], union_rank=union_rank,
        minimum_sketch_coverage=min(scores), mean_sketch_coverage=sum(scores)/len(scores),
        tasks=entries, selections=choices,
        note='Coverage on normalized frozen low-rank snapshots, not current old-task activations')
