import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.freq_decomp import (
    multilevel_dwt,
    freq_band_energy_from_bands,
    freq_band_energy_from_image,
    num_subbands,
)


# ============================================================
# 基础复原损失
# ============================================================

class L1Loss(nn.Module):
    """标准 L1 损失。"""
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, gt):
        return self.loss(pred, gt)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier 损失，常用于图像复原。
    loss = sqrt((pred - gt)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    感知损失。
    优先使用 LPIPS（若已安装），否则回退到 L1。
    输入范围：[0, 1]。
    """
    def __init__(self, net='vgg', device='cuda'):
        super().__init__()
        self.device = device
        self.use_lpips = False
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net).to(device)
            self.use_lpips = True
        except Exception:
            self.lpips = None

    def forward(self, pred, gt):
        if self.use_lpips:
            p = pred * 2 - 1
            g = gt * 2 - 1
            return self.lpips(p, g).mean()
        else:
            return F.l1_loss(pred, gt)


# ============================================================
# 正交损失 / 频带损失 / 共享损失
# ============================================================

class OrthogonalLoss(nn.Module):
    """
    正交损失：惩罚当前任务低秩基 B_t 与旧任务梯度空间 M 的重叠。
    B_t: [r, d]
    M_list: list of [d, k] 或单个 [d, k]
    """
    def forward(self, B_t, M_list):
        if M_list is None:
            return torch.tensor(0.0, device=B_t.device)
        if not isinstance(M_list, (list, tuple)):
            M_list = [M_list]
        loss = 0.0
        count = 0
        for m in M_list:
            if m is None or m.numel() == 0:
                continue
            proj = B_t @ m  # [r, k]
            loss = loss + proj.pow(2).mean()
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=B_t.device)
        return loss / count


class FrequencyComplementarityLoss(nn.Module):
    """
    频带互补损失：鼓励当前任务与旧任务在 DWT 子带能量分布上互补。
    p_t:   当前任务各子带能量占比，list 或 tensor
    p_old: 旧任务各子带能量占比，list 或 tensor
    重叠度越小越好。
    """
    def forward(self, p_t, p_old):
        if p_old is None:
            return torch.tensor(0.0, device=p_t.device if torch.is_tensor(p_t) else 'cpu')
        if not torch.is_tensor(p_t):
            p_t = torch.tensor(p_t, dtype=torch.float32)
        if not torch.is_tensor(p_old):
            p_old = torch.tensor(p_old, dtype=torch.float32)
        # 对齐长度
        min_len = min(p_t.numel(), p_old.numel())
        p_t = p_t[:min_len]
        p_old = p_old[:min_len]
        overlap = torch.minimum(p_t, p_old).sum()
        return overlap


class FrequencySelectionRegularization(nn.Module):
    """
    频带选择正则化：对频带重要性向量做稀疏 / 熵正则。
    freq_importance: [bands] 或 [C, bands]
    """
    def __init__(self, mode='entropy', weight=0.01):
        super().__init__()
        self.mode = mode
        self.weight = weight

    def forward(self, freq_importance):
        if freq_importance is None or freq_importance.numel() == 0:
            return torch.tensor(0.0, device='cpu')
        if self.mode == 'entropy':
            p = freq_importance / (freq_importance.sum() + 1e-12)
            entropy = -(p * torch.log(p + 1e-12)).sum()
            return -self.weight * entropy
        elif self.mode == 'sparse':
            return self.weight * freq_importance.abs().sum()
        else:
            return torch.tensor(0.0, device=freq_importance.device)


class ShareLoss(nn.Module):
    """
    共享损失：约束共享子空间基 B_share 保持正交、低冗余。
    B_share: [r_share, d]
    """
    def forward(self, B_share):
        if B_share is None or B_share.numel() == 0:
            return torch.tensor(0.0, device='cpu')
        gram = B_share @ B_share.T
        I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - I).pow(2).mean()


# ============================================================
# 总损失
# ============================================================

class TotalLoss(nn.Module):
    """
    总损失 = w_l1 * L1 + w_perc * Perceptual
    正交 / 频带互补 / 频带选择 / 共享损失由 trainer 中的辅助函数计算后额外加入。
    """
    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg['train']['loss']
        self.w_l1 = loss_cfg.get('l1', 1.0)
        self.w_perc = loss_cfg.get('perceptual', 0.1)
        self.w_orth = loss_cfg.get('orth', 0.01)
        self.w_freq = loss_cfg.get('freq', 0.01)
        self.w_share = loss_cfg.get('share', 0.001)
        self.w_freq_sel = loss_cfg.get('freq_sel', 0.001)

        self.l1 = L1Loss()
        self.perceptual = PerceptualLoss(device=cfg['experiment'].get('device', 'cuda'))
        self.freq_comp = FrequencyComplementarityLoss()
        self.freq_sel = FrequencySelectionRegularization(mode='entropy')
        self.share = ShareLoss()

    def forward(self, pred, gt):
        loss = self.w_l1 * self.l1(pred, gt)
        loss = loss + self.w_perc * self.perceptual(pred, gt)
        return loss


# ============================================================
# 遍历模型计算辅助损失
# ============================================================

def compute_orthogonal_loss(model):
    """
    遍历模型中的 ODRLoRAConv2d 模块，计算当前任务 B_t 与旧任务梯度空间的正交损失。
    支持按空间块和子带分别计算。
    """
    total = 0.0
    count = 0
    device = None
    for m in model.modules():
        if not hasattr(m, 'task_B') or not hasattr(m, 'dual_gpms'):
            continue
        idx = getattr(m, 'current_task_idx', -1)
        if idx <= 0 or idx >= len(m.task_B):
            continue
        B_t = m.task_B[idx]
        if B_t is None:
            continue
        device = B_t.device
        for key, gpm in m.dual_gpms.items():
            if gpm.basis is not None:
                proj = B_t @ gpm.basis
                total = total + proj.pow(2).mean()
                count += 1
    if count == 0:
        if device is None:
            device = 'cpu'
        return torch.tensor(0.0, device=device)
    return total / count


def compute_share_loss(model):
    """
    遍历模型中的共享子空间基 B_share，计算正交约束损失。
    """
    total = 0.0
    count = 0
    device = None
    for m in model.modules():
        if not hasattr(m, 'B_share'):
            continue
        B_share = getattr(m, 'B_share', None)
        if B_share is None or B_share.numel() == 0:
            continue
        device = B_share.device
        gram = B_share @ B_share.T
        I = torch.eye(gram.size(0), device=device, dtype=gram.dtype)
        total = total + (gram - I).pow(2).mean()
        count += 1
    if count == 0:
        if device is None:
            device = 'cpu'
        return torch.tensor(0.0, device=device)
    return total / count


def compute_frequency_loss(model, current_task_idx, freq_bands=7):
    """
    遍历模型中的频带能量统计，计算当前任务与旧任务的频带互补损失。
    若模型未保存频带能量，则返回 0。
    频带能量应分别在 calibrate 和 update_gpm 中保存为
    module.freq_energy_current 和 module.freq_energy_old。
    """
    device = None
    total = 0.0
    count = 0
    for m in model.modules():
        p_t = getattr(m, 'freq_energy_current', None)
        p_old = getattr(m, 'freq_energy_old', None)
        if p_t is None or p_old is None:
            continue
        if not torch.is_tensor(p_t):
            p_t = torch.tensor(p_t, dtype=torch.float32, device=device)
        if not torch.is_tensor(p_old):
            p_old = torch.tensor(p_old, dtype=torch.float32, device=device)
        if device is None:
            device = p_t.device
        min_len = min(p_t.numel(), p_old.numel())
        p_t = p_t[:min_len]
        p_old = p_old[:min_len]
        overlap = torch.minimum(p_t, p_old).sum()
        total = total + overlap
        count += 1
    if count == 0:
        if device is None:
            device = 'cpu'
        return torch.tensor(0.0, device=device)
    return total / count


def compute_frequency_selection_loss(model):
    """
    遍历模型中的频带重要性向量，计算频带选择正则化损失。
    """
    total = 0.0
    count = 0
    device = None
    for m in model.modules():
        freq_imp = getattr(m, 'freq_importance', None)
        if freq_imp is None or freq_imp.numel() == 0:
            continue
        device = freq_imp.device
        p = freq_imp / (freq_imp.sum() + 1e-12)
        entropy = -(p * torch.log(p + 1e-12)).sum()
        total = total + (-entropy)
        count += 1
    if count == 0:
        if device is None:
            device = 'cpu'
        return torch.tensor(0.0, device=device)
    return total / count


# ============================================================
# 导出
# ============================================================

__all__ = [
    "L1Loss",
    "CharbonnierLoss",
    "PerceptualLoss",
    "OrthogonalLoss",
    "FrequencyComplementarityLoss",
    "FrequencySelectionRegularization",
    "ShareLoss",
    "TotalLoss",
    "compute_orthogonal_loss",
    "compute_frequency_loss",
    "compute_share_loss",
    "compute_frequency_selection_loss",
]