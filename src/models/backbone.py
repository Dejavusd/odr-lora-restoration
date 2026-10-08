import torch
import torch.nn as nn
import torch.nn.functional as F

from .odr_lora import ODRLoRAConv2d


class SimpleGate(nn.Module):
    """NAFNet 中的 SimpleGate：将通道一分为二，相乘。"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention。"""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return x * y


class NAFBlock(nn.Module):
    """
    NAFNet 基础块，所有卷积替换为 ODRLoRAConv2d。
    """
    def __init__(
        self,
        channels,
        kernel_size=3,
        share_rank=8,
        private_rank=8,
        layer_scale=1.0,
        rank_gamma=1.5,
        rank_min=4,
        rank_max=32,
        spatial_blocks=(2, 2),
        spatial_sample_ratio=0.1,
        freq_level=2,
        adaptive_lambda_min=0.5,
        adaptive_lambda_max=1.0,
        skip_orthogonal=True,
        share_subspace=True,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = ODRLoRAConv2d(
            channels, channels * 2, kernel_size,
            share_rank=share_rank,
            private_rank=private_rank,
            layer_scale=layer_scale,
            rank_gamma=rank_gamma,
            rank_min=rank_min,
            rank_max=rank_max,
            spatial_blocks=spatial_blocks,
            spatial_sample_ratio=spatial_sample_ratio,
            freq_level=freq_level,
            adaptive_lambda_min=adaptive_lambda_min,
            adaptive_lambda_max=adaptive_lambda_max,
            skip_orthogonal=skip_orthogonal,
            share_subspace=share_subspace,
            **kwargs,
        )
        self.gate = SimpleGate()
        self.conv2 = ODRLoRAConv2d(
            channels, channels, kernel_size,
            share_rank=share_rank,
            private_rank=private_rank,
            layer_scale=layer_scale,
            rank_gamma=rank_gamma,
            rank_min=rank_min,
            rank_max=rank_max,
            spatial_blocks=spatial_blocks,
            spatial_sample_ratio=spatial_sample_ratio,
            freq_level=freq_level,
            adaptive_lambda_min=adaptive_lambda_min,
            adaptive_lambda_max=adaptive_lambda_max,
            skip_orthogonal=skip_orthogonal,
            share_subspace=share_subspace,
            **kwargs,
        )
        self.sca = SCA(channels)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, task_idx=None):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x, task_idx)
        x = self.gate(x)
        x = self.conv2(x, task_idx)
        x = self.sca(x)
        x = x * self.beta + residual
        return x


class NAFNet(nn.Module):
    """
    完整 NAFNet 风格骨干，所有卷积替换为 ODRLoRAConv2d。
    编码器-解码器 + 跳跃连接。
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        num_blocks=(2, 4, 8, 4, 2),
        kernel_size=3,
        share_rank=8,
        private_rank=8,
        rank_gamma=1.5,
        rank_min=4,
        rank_max=32,
        spatial_blocks=(2, 2),
        spatial_sample_ratio=0.1,
        freq_level=2,
        adaptive_lambda_min=0.5,
        adaptive_lambda_max=1.0,
        skip_orthogonal=True,
        share_subspace=True,
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chs = [base_channels * (2 ** i) for i in range(self.num_levels)]

        # 输入卷积
        self.input_conv = ODRLoRAConv2d(
            in_channels, chs[0], kernel_size,
            share_rank=share_rank,
            private_rank=private_rank,
            layer_scale=1.0,
            rank_gamma=rank_gamma,
            rank_min=rank_min,
            rank_max=rank_max,
            spatial_blocks=spatial_blocks,
            spatial_sample_ratio=spatial_sample_ratio,
            freq_level=freq_level,
            adaptive_lambda_min=adaptive_lambda_min,
            adaptive_lambda_max=adaptive_lambda_max,
            skip_orthogonal=skip_orthogonal,
            share_subspace=share_subspace,
            **kwargs,
        )

        # 编码器
        for i in range(self.num_levels):
            blocks = nn.ModuleList([
                NAFBlock(
                    chs[i], kernel_size,
                    share_rank=share_rank,
                    private_rank=private_rank,
                    layer_scale=1.0 + i * 0.2,
                    rank_gamma=rank_gamma,
                    rank_min=rank_min,
                    rank_max=rank_max,
                    spatial_blocks=spatial_blocks,
                    spatial_sample_ratio=spatial_sample_ratio,
                    freq_level=freq_level,
                    adaptive_lambda_min=adaptive_lambda_min,
                    adaptive_lambda_max=adaptive_lambda_max,
                    skip_orthogonal=skip_orthogonal,
                    share_subspace=share_subspace,
                )
                for _ in range(num_blocks[i])
            ])
            self.enc_blocks.append(blocks)
            if i < self.num_levels - 1:
                self.downs.append(
                    nn.Conv2d(chs[i], chs[i+1], kernel_size=2, stride=2)
                )

        # 解码器
        for i in range(self.num_levels - 1, -1, -1):
            if i < self.num_levels - 1:
                self.ups.append(
                    nn.ConvTranspose2d(chs[i+1], chs[i], kernel_size=2, stride=2)
                )
            blocks = nn.ModuleList([
                NAFBlock(
                    chs[i], kernel_size,
                    share_rank=share_rank,
                    private_rank=private_rank,
                    layer_scale=1.0 + i * 0.2,
                    rank_gamma=rank_gamma,
                    rank_min=rank_min,
                    rank_max=rank_max,
                    spatial_blocks=spatial_blocks,
                    spatial_sample_ratio=spatial_sample_ratio,
                    freq_level=freq_level,
                    adaptive_lambda_min=adaptive_lambda_min,
                    adaptive_lambda_max=adaptive_lambda_max,
                    skip_orthogonal=skip_orthogonal,
                    share_subspace=share_subspace,
                )
                for _ in range(num_blocks[i])
            ])
            self.dec_blocks.append(blocks)

        # 输出卷积
        self.output_conv = ODRLoRAConv2d(
            chs[0], out_channels, kernel_size,
            share_rank=share_rank,
            private_rank=private_rank,
            layer_scale=1.0,
            rank_gamma=rank_gamma,
            rank_min=rank_min,
            rank_max=rank_max,
            spatial_blocks=spatial_blocks,
            spatial_sample_ratio=spatial_sample_ratio,
            freq_level=freq_level,
            adaptive_lambda_min=adaptive_lambda_min,
            adaptive_lambda_max=adaptive_lambda_max,
            skip_orthogonal=skip_orthogonal,
            share_subspace=share_subspace,
            **kwargs,
        )

    def forward(self, x, task_idx=None):
        x = self.input_conv(x, task_idx)
        skips = []
        for i in range(self.num_levels):
            for block in self.enc_blocks[i]:
                x = block(x, task_idx)
            skips.append(x)
            if i < self.num_levels - 1:
                x = self.downs[i](x)

        for i in range(self.num_levels - 1, -1, -1):
            if i < self.num_levels - 1:
                x = self.ups[self.num_levels - 1 - i - 1](x)
            x = x + skips[i]
            for block in self.dec_blocks[self.num_levels - 1 - i]:
                x = block(x, task_idx)

        x = self.output_conv(x, task_idx)
        return x

    # ---------- 任务管理接口 ----------
    def add_task(self):
        for module in self.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.add_task()

    def set_current_task(self, idx):
        for module in self.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.set_current_task(idx)

    def freeze_base_and_old(self, current_idx):
        for module in self.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.freeze_base_and_old(current_idx)

    # ---------- 校准辅助：捕获每个 ODRLoRAConv2d 的真实输入 ----------
    def _capture_layer_inputs(self, x):
        """跑一次前向，用 hook 捕获每个 ODRLoRAConv2d 的真实输入。"""
        inputs = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                inputs[name] = inp[0].detach()
            return hook

        for name, module in self.named_modules():
            if isinstance(module, ODRLoRAConv2d):
                hooks.append(module.register_forward_hook(make_hook(name)))

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _ = self.forward(x)
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                self.train()
        return inputs

    def calibrate(self, x, current_idx, task_similarity=0.0):
        # 每层用各自的真实输入做校准，避免把 3 通道原图传给 16 通道的层
        inputs = self._capture_layer_inputs(x)
        for name, module in self.named_modules():
            if isinstance(module, ODRLoRAConv2d) and name in inputs:
                module.calibrate(inputs[name], current_idx, task_similarity)

    def update_gpm(self, x, current_idx):
        # 同样每层用各自的真实输入
        inputs = self._capture_layer_inputs(x)
        for name, module in self.named_modules():
            if isinstance(module, ODRLoRAConv2d) and name in inputs:
                module.update_gpm(inputs[name], current_idx)

    def merge_task(self, idx):
        for module in self.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.merge_task(idx)