import torch
import torch.nn as nn

from .backbone import NAFNet, NAFBlock, PixelChannelNorm, BoundedGate
from .odr_lora import ODRLoRAConv2d


class RestorationNet(nn.Module):
    """
    完整图像复原网络，骨干为 NAFNet，所有卷积替换为 ODRLoRAConv2d。
    """
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        o = cfg['odr_lora']
        self.new_task_private_rank_base = m.get('new_task_private_rank_base')
        architecture = m.get('architecture', 'legacy')
        if architecture not in ('legacy', 'stable_v1'):
            raise ValueError(f'Unknown architecture: {architecture}')

        # 频带级别：1 -> 4 子带，2 -> 7 子带
        freq_level = o.get('freq_level', 2)

        self.backbone = NAFNet(
            in_channels=3,
            out_channels=3,
            base_channels=m['base_channels'],
            num_blocks=m['num_blocks'],
            kernel_size=m['kernel_size'],
            share_rank=m['share_rank'],
            private_rank=m['private_rank_base'],
            rank_gamma=m['rank_gamma'],
            rank_min=m['rank_min'],
            rank_max=m['rank_max'],
            spatial_blocks=tuple(o['spatial_blocks']),
            spatial_sample_ratio=o['spatial_sample_ratio'],
            freq_level=freq_level,
            freq_decomp=o.get('freq_decomp', 'dwt'),
            adaptive_lambda_min=o['adaptive_lambda_min'],
            adaptive_lambda_max=o['adaptive_lambda_max'],
            skip_orthogonal=o['skip_orthogonal'],
            share_subspace=o['share_subspace'],
        )
        if architecture == 'stable_v1':
            for block in self.backbone.modules():
                if isinstance(block, NAFBlock):
                    channels = block.beta.shape[1]
                    block.norm1 = PixelChannelNorm(channels)
                    block.norm2 = PixelChannelNorm(channels)
                    block.gate = BoundedGate()
                    block.sca.bounded = True
                    block.bounded_residual = True
        for module in self.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.full_update_protection = o.get('full_update_protection', False)
                module.basis_construction = o.get('basis_construction', 'legacy')
                module.basis_rank_tolerance = float(o.get('basis_rank_tolerance', 1e-5))
                if (module.basis_construction not in ('legacy', 'projected_svd')
                        or module.basis_rank_tolerance <= 0
                        or (module.basis_construction == 'projected_svd'
                            and not module.full_update_protection)):
                    raise ValueError('Invalid basis construction or protection setting')
                module.gpm_statistics = o.get('gpm_statistics', 'frequency')
                module.gpm_update_rule = o.get('gpm_update_rule', 'legacy')
                module.gpm_sample_columns = int(o.get('gpm_sample_columns', 512))
                capacity = int(o.get('gpm_max_basis', m['rank_max']))
                if (module.gpm_statistics not in ('frequency', 'raw')
                        or module.gpm_update_rule not in ('legacy','bounded_svd','bounded_preserving','bounded_balanced')
                        or capacity < 1 or module.gpm_sample_columns < 1):
                    raise ValueError('Invalid GPM statistics, capacity or sample count')
                module.dual_gpms['global'].max_basis = min(capacity, module.d)

    def forward(self, x, task_idx=None):
        return self.backbone(x, task_idx)

    # ---------- 任务管理接口 ----------
    def add_task(self, private_rank_base=None):
        if private_rank_base is None and len(self.backbone.input_conv.task_A) > 0:
            private_rank_base = self.new_task_private_rank_base
        self.backbone.add_task(private_rank_base=private_rank_base)

    def set_current_task(self, idx):
        self.backbone.set_current_task(idx)

    def freeze_base_and_old(self, idx):
        self.backbone.freeze_base_and_old(idx)

    def calibrate(self, x, idx, sim=0.0):
        self.backbone.calibrate(x, idx, sim)

    def update_gpm(self, x, idx):
        self.backbone.update_gpm(x, idx)

    def merge_task(self, idx):
        self.backbone.merge_task(idx)

    def prepare_for_inference(self):
        self.backbone.prepare_for_inference()
