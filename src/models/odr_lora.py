import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_gpm import DualGPM
from .spatial_block import (
    im2col_block,
    spatial_aware_subsample,
    spatial_subband_im2col,
    split_spatial_blocks,
    num_subbands,
)
from src.utils.svd_utils import topk_right_singular


class ODRLoRAConv2d(nn.Module):
    """
    ODR-LoRA 卷积层：共享基座 + 动态低秩分支 + 空间/频率/尺度感知正交。

    权重形状：[C_out, C_in, K, K]
    内部展平为 [C_out, d]，d = C_in * K * K。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=True,
        share_rank=8,
        private_rank=8,
        spatial_blocks=(2, 2),
        spatial_sample_ratio=0.1,
        freq_level=2,                # 1 -> 4 子带，2 -> 7 子带
        freq_decomp='dwt',
        adaptive_lambda_min=0.5,
        adaptive_lambda_max=1.0,
        skip_orthogonal=True,
        is_skip_fusion=False,
        share_subspace=True,
        layer_scale=1.0,
        rank_gamma=1.5,
        rank_min=4,
        rank_max=32,
        dual_gpm_eps=0.99,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        requested_share_rank = share_rank
        self.share_rank = share_rank if share_subspace else 0
        self.spatial_blocks = spatial_blocks
        self.spatial_sample_ratio = spatial_sample_ratio
        self.freq_level = freq_level
        self.freq_decomp = freq_decomp
        self.freq_bands = (
            num_subbands(level=freq_level) if freq_decomp != 'none' else 1
        )
        self.adaptive_lambda_min = adaptive_lambda_min
        self.adaptive_lambda_max = adaptive_lambda_max
        self.skip_orthogonal = skip_orthogonal
        self.is_skip_fusion = is_skip_fusion
        self.share_subspace = share_subspace
        self.rank_gamma = rank_gamma
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.layer_scale = layer_scale
        self.dual_gpm_eps = dual_gpm_eps

        # 根据尺度计算 private rank
        self.private_rank = self._compute_rank(private_rank)
        if not share_subspace:
            # Keep the total rank budget comparable in the no-sharing
            # ablation instead of leaving zero-valued shared rows in B.
            self.private_rank = min(
                self.rank_max, self.private_rank + requested_share_rank
            )
        self.total_rank = self.share_rank + self.private_rank
        d = in_channels * kernel_size * kernel_size
        self.d = d

        # 基座权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 共享子空间基 B_share: [share_rank, d]
        if share_subspace:
            self.register_buffer(
                'B_share',
                torch.zeros(share_rank, d)
            )
            self.register_buffer('share_initialized', torch.tensor(False))
        else:
            self.B_share = None
            self.register_buffer('share_initialized', torch.tensor(False))

        # 任务专属分支：每个任务一个 (A_t, B_t)
        # A_t: [C_out, total_rank]
        # B_t: [total_rank, d]，前 share_rank 行来自 B_share，后 private_rank 行是私有基
        self.task_A = nn.ParameterList()
        self.task_B = nn.ParameterList()
        self.task_ids = []

        # Keep one bounded protected subspace per layer. Spatial blocks and
        # frequency bands improve sampling coverage; giving every (g, f) pair
        # a full d-dimensional GPM makes memory grow by grid_h*grid_w*bands.
        self.dual_gpms = nn.ModuleDict({
            'global': DualGPM(
                dim=d,
                eps=dual_gpm_eps,
                max_basis=min(rank_max, d),
                use_orthogonal_complement=False,
            )
        })

        self.current_task_idx = -1
        # Opt-in ablation: constrain the complete update, including shared rows.
        # Store the projected B itself so forward, checkpoint and merge agree.
        self.full_update_protection = False
        self.basis_construction = 'legacy'
        self.basis_rank_tolerance = 1e-5
        self.gpm_statistics = 'frequency'
        self.gpm_sample_columns = 512
        self._merged_tasks = set()
        self._init_weights()

    def _compute_rank(self, base_rank):
        r = base_rank * (self.layer_scale ** self.rank_gamma)
        r = int(round(r))
        return max(self.rank_min, min(self.rank_max, r))

    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.weight, mode='fan_out', nonlinearity='relu'
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.B_share is not None:
            nn.init.orthogonal_(self.B_share)

    def _extract_statistics(self, x):
        if self.freq_decomp != 'none':
            return spatial_subband_im2col(
                x,
                *self.spatial_blocks,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                level=self.freq_level,
                sample_ratio=self.spatial_sample_ratio,
            )

        pairs = []
        for g, block in enumerate(split_spatial_blocks(x, *self.spatial_blocks)):
            H = im2col_block(
                block, self.kernel_size,
                stride=self.stride, padding=self.padding,
            )
            batch, dim, columns = H.shape
            H = H.permute(1, 0, 2).reshape(dim, batch * columns)
            H = spatial_aware_subsample(H, ratio=self.spatial_sample_ratio)
            pairs.append((g, 0, H))
        return pairs

    # ---------- 任务管理 ----------
    def add_task(self, private_rank=None):
        """新增一个任务分支，返回任务索引。"""
        d = self.d
        total_rank = self.total_rank if private_rank is None else self.share_rank + int(private_rank)
        if total_rank < self.share_rank:
            raise ValueError('private_rank must be nonnegative')
        ref = self.weight   # 用基座权重的 device/dtype 作为参考
        A = nn.Parameter(torch.zeros(
            self.out_channels, total_rank,
            device=ref.device, dtype=ref.dtype,
        ))
        B = nn.Parameter(torch.zeros(
            total_rank, d,
            device=ref.device, dtype=ref.dtype,
        ))
        # 前 share_rank 行从 B_share 复制（若已初始化）
        if self.B_share is not None and bool(self.share_initialized):
            B.data[:self.share_rank] = self.B_share.data
        self.task_A.append(A)
        self.task_B.append(B)
        idx = len(self.task_A) - 1
        self.task_ids.append(idx)
        return idx

    def set_current_task(self, idx):
        self.current_task_idx = idx

    def freeze_base_and_old(self, current_idx):
        """冻结基座和旧任务分支，只训练当前任务 A。"""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.B_share is not None:
            self.B_share.requires_grad = False
        for i, (A, B) in enumerate(zip(self.task_A, self.task_B)):
            if i < current_idx:
                A.requires_grad = False
                B.requires_grad = False
            elif i == current_idx:
                A.requires_grad = True
                B.requires_grad = False
            else:
                A.requires_grad = False
                B.requires_grad = False

    # ---------- 校准：计算 B_t ----------
    @torch.no_grad()
    def calibrate(self, x, current_idx, task_similarity=0.0):
        """
        用当前任务输入 x 校准该层的 B_t。
        x: [B, C_in, H, W]
        current_idx: 当前任务索引
        task_similarity: 与旧任务最大相似度 s_t
        """
        self.eval()
        if current_idx >= len(self.task_B):
            return
        B = self.task_B[current_idx]
        grid_h, grid_w = self.spatial_blocks

        # 1. 空间分块 + 多级 DWT + im2col + 子采样
        pairs = self._extract_statistics(x)
        # pairs: list of (g, f, H_sub)

        if len(pairs) == 0:
            return

        # 2. Spatial/frequency identity is used for coverage; all samples form
        # one layer-level activation matrix so the resulting convolution basis
        # remains translation shared and mergeable.
        H_all = torch.cat([H_sub for _, _, H_sub in pairs], dim=1)
        if H_all.numel() == 0 or H_all.shape[1] == 0:
            return

        # 4. Initialize the shared basis before deriving the private residual.
        if self.B_share is not None and not bool(self.share_initialized):
            V_share = topk_right_singular(H_all.T.float(), self.share_rank, method='random')
            self.B_share.data.copy_(V_share.to(self.B_share))
            self.share_initialized.fill_(True)

        if self.basis_construction == 'projected_svd':
            if not self.full_update_protection:
                raise ValueError('projected_svd requires full_update_protection')
            self._calibrate_projected(H_all, current_idx)
            return

        if self.B_share is not None and bool(self.share_initialized):
            Bs = self.B_share  # [r_share, d]
            B.data[:self.share_rank] = Bs.data
            proj = Bs.T @ (Bs @ H_all)
            H_all_private = H_all - proj
        else:
            H_all_private = H_all

        # 5. Protect the old activation subspace after removing the shared
        # component. This follows (I - alpha M M^T)(I - B_s^T B_s)H.
        gpm = self.dual_gpms['global']
        protect = not (self.is_skip_fusion and not self.skip_orthogonal)
        if gpm.basis is not None and protect:
            alpha = self.adaptive_lambda_min + (
                self.adaptive_lambda_max - self.adaptive_lambda_min
            ) * (1.0 - task_similarity)
            H_all_private = (
                H_all_private - alpha * gpm.project_in(H_all_private)
            )

        # 6. SVD 取 top-private_rank 右奇异向量
        H_all_private = H_all_private.float()
        if H_all_private.shape[1] == 0:
            return
        private_rank = B.shape[0] - self.share_rank
        if private_rank == 0:
            if self.full_update_protection:
                self.protect_task_basis(current_idx)
            return
        V = topk_right_singular(H_all_private.T, private_rank, method='random')
        # V: [private_rank, d]
        B.data[self.share_rank:].zero_()
        B.data[self.share_rank:self.share_rank + V.shape[0]] = V.to(B)
        if self.full_update_protection:
            self.protect_task_basis(current_idx)

    @torch.no_grad()
    def _calibrate_projected(self, H, task_idx):
        """Construct unit bases INSIDE the old GPM nullspace before selection.

        Keep only non-negligible singular directions. Unused allocated rows
        stay zero; never normalize numerical nullspace residue into a vector.
        Private directions are also orthogonal to the effective shared span.
        """
        B = self.task_B[task_idx]
        M = self.dual_gpms['global'].basis

        def project_rows(rows):
            if M is not None:
                for _ in range(2):
                    rows = rows - (rows @ M) @ M.T
            return rows

        def unit_rows(matrix, count):
            if count <= 0 or matrix.numel() == 0:
                return matrix.new_empty((0, self.d))
            _, values, vectors = torch.linalg.svd(matrix.float(), full_matrices=False)
            threshold = max(1e-6, float(values[0]) * self.basis_rank_tolerance)
            rank = min(count, int((values > threshold).sum()))
            return vectors[:rank].to(B)

        shared = B.new_empty((0, self.d))
        if self.B_share is not None:
            shared = unit_rows(project_rows(self.B_share.float()), self.share_rank)
        residual = project_rows(H.T.float())
        if shared.numel():
            for _ in range(2):
                residual = residual - (residual @ shared.T) @ shared
                residual = project_rows(residual)
        private = unit_rows(residual, B.shape[0] - self.share_rank)
        B.zero_()
        B[:shared.shape[0]].copy_(shared)
        B[self.share_rank:self.share_rank + private.shape[0]].copy_(private)
        self.protect_task_basis(task_idx)

    @torch.no_grad()
    def protect_task_basis(self, task_idx):
        """Make (A @ B) @ M = 0 for every A, with B frozen during training.

        Apply once after calibration, using the OLD GPM. Do not project against
        a GPM updated after training: that would change the learned function.
        No row normalization follows projection, including nearly-zero rows.
        """
        basis = self.dual_gpms['global'].basis
        if basis is not None:
            B = self.task_B[task_idx]
            # A second pass removes floating-point residuals, particularly in
            # small layers whose protected subspace nearly spans all inputs.
            for _ in range(2):
                B.copy_(B - (B @ basis) @ basis.T)

    @torch.no_grad()
    def update_gpm(self, x, current_idx):
        """训练完当前任务后，用当前任务输入更新 DualGPM。"""
        self.eval()
        if self.is_skip_fusion and not self.skip_orthogonal:
            return
        if self.gpm_statistics == 'raw':
            H = self.raw_input_matrix(x)
            self.dual_gpms['global'].update_preserving(H)
            return
        grid_h, grid_w = self.spatial_blocks
        pairs = self._extract_statistics(x)
        matrices = [H_sub for _, _, H_sub in pairs if H_sub.numel() > 0]
        if matrices:
            self.dual_gpms['global'].update(torch.cat(matrices, dim=1))

    def raw_input_matrix(self, x):
        """Bounded, uniformly sampled patches in actual convolution coordinates."""
        H = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        H = H.permute(1, 0, 2).reshape(self.d, -1)
        count = min(self.gpm_sample_columns, H.shape[1])
        indices = torch.linspace(0, H.shape[1] - 1, count, device=H.device).long()
        return H[:, indices]

    # ---------- 前向 ----------
    def forward(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task_idx
        # 基座卷积
        out = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )
        # 加上任务分支
        if task_idx >= 0 and task_idx < len(self.task_A):
            A = self.task_A[task_idx]  # [C_out, total_rank]
            B = self.task_B[task_idx]  # [total_rank, d]
            W_task = A @ B  # [C_out, d]
            W_task = W_task.view(self.out_channels, self.in_channels,
                                 self.kernel_size, self.kernel_size)
            out = out + F.conv2d(x, W_task, None, stride=self.stride, padding=self.padding)
        return out

    def merge_task(self, task_idx, clear_adapter=True):
        """把任务分支合并进基座权重（推理时使用）。

        Clearing A after the merge preserves numerical equivalence and avoids
        accidentally applying the same update twice.
        """
        with torch.no_grad():
            if (task_idx >= 0 and task_idx < len(self.task_A)
                    and task_idx not in self._merged_tasks):
                A = self.task_A[task_idx]
                B = self.task_B[task_idx]
                W_task = A @ B
                self.weight.data += W_task.view_as(self.weight)
                if clear_adapter:
                    A.zero_()
                self._merged_tasks.add(task_idx)
                if self.current_task_idx == task_idx:
                    self.current_task_idx = -1

    def prepare_for_inference(self):
        """Drop training-only adapters and activation statistics after merge."""
        self.task_A = nn.ParameterList()
        self.task_B = nn.ParameterList()
        self.task_ids = []
        self.current_task_idx = -1
        self._merged_tasks.clear()
        for gpm in self.dual_gpms.values():
            gpm.basis = None
            gpm.singular_values = None
