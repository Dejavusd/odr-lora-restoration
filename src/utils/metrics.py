"""
评估指标：PSNR / SSIM / LPIPS。

输入约定：
    pred, gt: torch.Tensor，形状 [B, C, H, W] 或 [C, H, W]
    数值范围：[0, 1]
"""

import torch
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# ============================================================
# 张量转 numpy
# ============================================================

def _to_numpy(tensor):
    """
    将 [B, C, H, W] 或 [C, H, W] 的 [0,1] 张量转为 HWC numpy 数组。
    若 batch > 1，取第一个样本。
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    # [C, H, W] -> [H, W, C]
    array = tensor.permute(1, 2, 0).numpy()
    return array


# ============================================================
# PSNR
# ============================================================

def compute_psnr(pred, gt):
    """
    计算 PSNR。
    pred, gt: [B, C, H, W] 或 [C, H, W]，范围 [0, 1]
    返回: float
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    if _HAS_SKIMAGE:
        return float(psnr_metric(gt_np, pred_np, data_range=1.0))

    # 回退实现
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


# ============================================================
# SSIM
# ============================================================

def compute_ssim(pred, gt):
    """
    计算 SSIM。
    pred, gt: [B, C, H, W] 或 [C, H, W]，范围 [0, 1]
    返回: float
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    if _HAS_SKIMAGE:
        return float(
            ssim_metric(
                gt_np, pred_np,
                channel_axis=2,
                data_range=1.0,
            )
        )

    # 回退实现：逐通道平均
    if pred_np.ndim == 2:
        pred_np = pred_np[..., None]
        gt_np = gt_np[..., None]
    scores = []
    for c in range(pred_np.shape[2]):
        p = pred_np[:, :, c]
        g = gt_np[:, :, c]
        mu_p = p.mean()
        mu_g = g.mean()
        var_p = p.var()
        var_g = g.var()
        cov = ((p - mu_p) * (g - mu_g)).mean()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu_p * mu_g + c1) * (2 * cov + c2)) / \
               ((mu_p ** 2 + mu_g ** 2 + c1) * (var_p + var_g + c2))
        scores.append(ssim)
    return float(np.mean(scores))


# ============================================================
# LPIPS
# ============================================================

class LPIPSMetric:
    """
    LPIPS 指标封装。
    若 lpips 未安装或加载失败，则回退到 L1 距离。
    输入范围：[0, 1]。
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.fn = None
        self.use_lpips = False
        try:
            import lpips
            self.fn = lpips.LPIPS(net='vgg').to(device)
            self.fn.eval()
            self.use_lpips = True
        except Exception:
            self.fn = None
            self.use_lpips = False

    @torch.no_grad()
    def __call__(self, pred, gt):
        """
        pred, gt: [B, C, H, W] 或 [C, H, W]，范围 [0, 1]
        返回: float
        """
        if self.use_lpips and self.fn is not None:
            p = pred.to(self.device)
            g = gt.to(self.device)
            if p.dim() == 3:
                p = p.unsqueeze(0)
            if g.dim() == 3:
                g = g.unsqueeze(0)
            # LPIPS 期望 [-1, 1]
            p = p * 2 - 1
            g = g * 2 - 1
            return float(self.fn(p, g).mean().item())

        # 回退：L1
        return float(torch.mean(torch.abs(pred - gt)).item())


# ============================================================
# 导出
# ============================================================

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "LPIPSMetric",
]