import torch
import torch.nn as nn

from .backbone import NAFNet


class RestorationNet(nn.Module):
    """
    完整图像复原网络，骨干为 NAFNet，所有卷积替换为 ODRLoRAConv2d。
    """
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        o = cfg['odr_lora']

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
            adaptive_lambda_min=o['adaptive_lambda_min'],
            adaptive_lambda_max=o['adaptive_lambda_max'],
            skip_orthogonal=o['skip_orthogonal'],
            share_subspace=o['share_subspace'],
        )

    def forward(self, x, task_idx=None):
        return self.backbone(x, task_idx)

    # ---------- 任务管理接口 ----------
    def add_task(self):
        self.backbone.add_task()

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