import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.restoration_net import RestorationNet
from src.data.datasets import build_task_loaders
from src.losses.losses import TotalLoss
from src.utils.metrics import compute_psnr, compute_ssim, LPIPSMetric
from src.utils.logger import get_logger, MetricLogger
from src.utils.checkpoint import save_checkpoint


# ============================================================
# 基类
# ============================================================

class BaseTrainer:
    """
    所有基线训练器的基类。
    提供通用的训练/评估/日志/保存逻辑。
    """

    def __init__(self, cfg, name="baseline"):
        self.cfg = cfg
        self.name = name
        self.device = torch.device(
            cfg['experiment'].get('device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
        self.output_dir = os.path.join(cfg['experiment']['output_dir'], name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = get_logger(name, self.output_dir)
        self.metric_logger = MetricLogger(os.path.join(self.output_dir, 'metrics.json'))

        self.task_order = [t['name'] for t in cfg['data']['tasks']]
        self.num_tasks = len(self.task_order)

        self.model = RestorationNet(cfg).to(self.device)
        self.loss_fn = TotalLoss(cfg).to(self.device)

    # ------------------------------------------------------------------
    # 优化器 / 调度器
    # ------------------------------------------------------------------
    def build_optimizer(self, params=None):
        if params is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg['train']['lr'],
            weight_decay=self.cfg['train']['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg['train']['epochs_per_task'], eta_min=1e-6
        )
        return optimizer, scheduler

    # ------------------------------------------------------------------
    # 单任务训练
    # ------------------------------------------------------------------
    def train_task(self, task_name, task_idx, epochs, train_loader=None):
        if train_loader is None:
            train_loader, val_loader = build_task_loaders(self.cfg, task_name)
        else:
            _, val_loader = build_task_loaders(self.cfg, task_name)

        optimizer, scheduler = self.build_optimizer()
        best_psnr = 0.0

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"{self.name} | {task_name} Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0

            for batch in pbar:
                x = batch['input'].to(self.device)
                gt = batch['gt'].to(self.device)
                pred = self.model(x, task_idx if self.use_task_id else None)
                loss = self.loss_fn(pred, gt)
                loss = loss + self.extra_loss()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg['train']['grad_clip']
                )
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            scheduler.step()
            self.logger.info(f"{task_name} Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}")

            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                metrics = self.evaluate_task(val_loader, task_idx)
                self.logger.info(
                    f"{task_name} Epoch {epoch+1} 验证: "
                    f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}"
                )
                self.metric_logger.log({
                    'task': task_name,
                    'task_idx': task_idx,
                    'epoch': epoch + 1,
                    **metrics
                })
                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    save_checkpoint(
                        self.model, optimizer, epoch + 1, task_idx,
                        os.path.join(self.output_dir, f'task_{task_idx}_best.pth')
                    )

        save_checkpoint(
            self.model, optimizer, epochs, task_idx,
            os.path.join(self.output_dir, f'task_{task_idx}_final.pth')
        )

    def extra_loss(self):
        """子类可重写，加入正则化项。"""
        return 0.0

    @property
    def use_task_id(self):
        """是否在前向时传入 task_idx。"""
        return False

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_task(self, loader, task_idx):
        self.model.eval()
        psnrs, ssims, lpipses = [], [], []
        lpips_fn = LPIPSMetric(self.device)

        for batch in loader:
            x = batch['input'].to(self.device)
            gt = batch['gt'].to(self.device)
            pred = self.model(x, task_idx if self.use_task_id else None)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            lpipses.append(lpips_fn(pred, gt))

        return {
            'psnr': float(np.mean(psnrs)),
            'ssim': float(np.mean(ssims)),
            'lpips': float(np.mean(lpipses)),
        }

    def evaluate_all(self):
        results = {}
        for idx, task_name in enumerate(self.task_order):
            _, val_loader = build_task_loaders(self.cfg, task_name)
            metrics = self.evaluate_task(val_loader, idx)
            results[task_name] = metrics
            self.logger.info(f"{task_name} 最终评估: {metrics}")
        avg_psnr = np.mean([r['psnr'] for r in results.values()])
        avg_ssim = np.mean([r['ssim'] for r in results.values()])
        self.logger.info(f"所有任务平均: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return results

    def run(self):
        raise NotImplementedError


# ============================================================
# 1. 联合训练（All-in-One）
# ============================================================

class JointTrainer(BaseTrainer):
    """
    所有任务数据混合，联合训练一个共享模型。
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="joint")
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def run(self):
        # 合并所有任务的训练集
        all_loaders = []
        for task_name in self.task_order:
            train_loader, _ = build_task_loaders(self.cfg, task_name)
            all_loaders.append(train_loader)

        # 简单轮流采样
        epochs = self.cfg['train']['epochs_per_task'] * self.num_tasks
        optimizer, scheduler = self.build_optimizer()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            count = 0
            for loader in all_loaders:
                for batch in loader:
                    x = batch['input'].to(self.device)
                    gt = batch['gt'].to(self.device)
                    pred = self.model(x, None)
                    loss = self.loss_fn(pred, gt)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg['train']['grad_clip']
                    )
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
            scheduler.step()
            self.logger.info(f"Joint Epoch {epoch+1}: loss={total_loss/max(count,1):.4f}")

        save_checkpoint(
            self.model, optimizer, epochs, -1,
            os.path.join(self.output_dir, 'joint_final.pth')
        )
        self.evaluate_all()


# ============================================================
# 2. 顺序微调（Sequential Fine-tuning）
# ============================================================

class SequentialTrainer(BaseTrainer):
    """
    按任务顺序微调整个模型，无参数隔离。
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="sequential")
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== Sequential 训练任务 {idx}: {task_name} =====")
            self.train_task(task_name, idx, epochs)
        self.evaluate_all()


# ============================================================
# 3. EWC
# ============================================================

class EWCTrainer(BaseTrainer):
    """
    Elastic Weight Consolidation。
    """

    def __init__(self, cfg, lambda_ewc=5000.0):
        super().__init__(cfg, name="ewc")
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.old_params = {}
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def _compute_fisher(self, loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in loader:
            x = batch['input'].to(self.device)
            gt = batch['gt'].to(self.device)
            self.model.zero_grad()
            pred = self.model(x, None)
            loss = self.loss_fn(pred, gt)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(loader)
        return fisher

    def extra_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.old_params:
                loss = loss + (self.fisher[n] * (p - self.old_params[n]).pow(2)).sum()
        return self.lambda_ewc * loss

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== EWC 训练任务 {idx}: {task_name} =====")
            train_loader, _ = build_task_loaders(self.cfg, task_name)
            self.train_task(task_name, idx, epochs, train_loader=train_loader)
            # 更新 Fisher 和旧参数
            self.fisher = self._compute_fisher(train_loader)
            self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.evaluate_all()


# ============================================================
# 4. SI（Synaptic Intelligence）
# ============================================================

class SITrainer(BaseTrainer):
    """
    Synaptic Intelligence。
    """

    def __init__(self, cfg, c_si=0.1, xi=1e-3):
        super().__init__(cfg, name="si")
        self.c_si = c_si
        self.xi = xi
        self.omega = {}
        self.old_params = {}
        self.w = {}
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== SI 训练任务 {idx}: {task_name} =====")
            # 初始化 w
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.w[n] = torch.zeros_like(p)
            train_loader, _ = build_task_loaders(self.cfg, task_name)
            self.train_task(task_name, idx, epochs, train_loader=train_loader)
            # 更新 omega
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.w:
                    delta = p.detach() - self.old_params.get(n, p.detach().clone())
                    self.omega[n] = self.omega.get(n, torch.zeros_like(p)) + self.w[n] / (delta.pow(2) + self.xi)
                    self.old_params[n] = p.detach().clone()
        self.evaluate_all()

    def extra_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.omega and n in self.old_params:
                loss = loss + (self.omega[n] * (p - self.old_params[n]).pow(2)).sum()
        return self.c_si * loss


# ============================================================
# 5. MAS（Memory Aware Synapses）
# ============================================================

class MASTrainer(BaseTrainer):
    """
    Memory Aware Synapses。
    """

    def __init__(self, cfg, lambda_mas=1.0):
        super().__init__(cfg, name="mas")
        self.lambda_mas = lambda_mas
        self.omega = {}
        self.old_params = {}
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def _compute_omega(self, loader):
        omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for batch in loader:
            x = batch['input'].to(self.device)
            self.model.zero_grad()
            pred = self.model(x, None)
            # 用输出 L2 范数作为重要性
            out_norm = pred.pow(2).sum()
            out_norm.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in omega:
                    omega[n] += p.grad.data.abs()
        for n in omega:
            omega[n] /= len(loader)
        return omega

    def extra_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.omega and n in self.old_params:
                loss = loss + (self.omega[n] * (p - self.old_params[n]).pow(2)).sum()
        return self.lambda_mas * loss

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== MAS 训练任务 {idx}: {task_name} =====")
            train_loader, _ = build_task_loaders(self.cfg, task_name)
            self.train_task(task_name, idx, epochs, train_loader=train_loader)
            self.omega = self._compute_omega(train_loader)
            self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.evaluate_all()


# ============================================================
# 6. LwF（Learning without Forgetting）
# ============================================================

class LwFTrainer(BaseTrainer):
    """
    Learning without Forgetting。
    用旧模型对当前任务输入的输出作为软目标。
    """

    def __init__(self, cfg, lambda_lwf=1.0, temperature=2.0):
        super().__init__(cfg, name="lwf")
        self.lambda_lwf = lambda_lwf
        self.temperature = temperature
        self.old_model = None
        self.use_task_id_flag = False

    @property
    def use_task_id(self):
        return False

    def extra_loss(self):
        if self.old_model is None:
            return 0.0
        # 需要在 train_task 中拿到当前 batch 的输入，这里简化：由外部设置
        return 0.0

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== LwF 训练任务 {idx}: {task_name} =====")
            train_loader, val_loader = build_task_loaders(self.cfg, task_name)
            # 保存旧模型
            if idx > 0:
                self.old_model = copy.deepcopy(self.model).to(self.device)
                self.old_model.eval()
                for p in self.old_model.parameters():
                    p.requires_grad = False

            optimizer, scheduler = self.build_optimizer()
            for epoch in range(epochs):
                self.model.train()
                pbar = tqdm(train_loader, desc=f"LwF | {task_name} Epoch {epoch+1}/{epochs}")
                for batch in pbar:
                    x = batch['input'].to(self.device)
                    gt = batch['gt'].to(self.device)
                    pred = self.model(x, None)
                    loss = self.loss_fn(pred, gt)
                    if self.old_model is not None:
                        with torch.no_grad():
                            old_pred = self.old_model(x, None)
                        # 软目标蒸馏
                        loss = loss + self.lambda_lwf * F.mse_loss(
                            F.log_softmax(pred / self.temperature, dim=1),
                            F.softmax(old_pred / self.temperature, dim=1)
                        ) * (self.temperature ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg['train']['grad_clip']
                    )
                    optimizer.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                scheduler.step()

            save_checkpoint(
                self.model, optimizer, epochs, idx,
                os.path.join(self.output_dir, f'task_{idx}_final.pth')
            )
        self.evaluate_all()


# ============================================================
# 7. 普通 LoRA 多任务
# ============================================================

class LoRATrainer(BaseTrainer):
    """
    普通 LoRA 多任务：基座冻结，每个任务新增一个 LoRA 分支，训练时只更新当前分支。
    需要模型支持 add_task / set_current_task / freeze_base_and_old。
    这里复用 RestorationNet 的 ODRLoRAConv2d 接口，但关闭正交、频带、共享等机制。
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="lora")
        self.use_task_id_flag = True

    @property
    def use_task_id(self):
        return True

    def run(self):
        epochs = self.cfg['train']['epochs_per_task']
        for idx, task_name in enumerate(self.task_order):
            self.logger.info(f"===== LoRA 训练任务 {idx}: {task_name} =====")
            train_loader, val_loader = build_task_loaders(self.cfg, task_name)
            self.model.add_task()
            self.model.set_current_task(idx)
            # 随机初始化 B，不用正交校准
            for m in self.model.modules():
                if hasattr(m, 'task_B') and len(m.task_B) > idx:
                    nn.init.kaiming_normal_(m.task_B[idx])
            self.model.freeze_base_and_old(idx)
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params, lr=self.cfg['train']['lr'],
                weight_decay=self.cfg['train']['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            for epoch in range(epochs):
                self.model.train()
                pbar = tqdm(train_loader, desc=f"LoRA | {task_name} Epoch {epoch+1}/{epochs}")
                for batch in pbar:
                    x = batch['input'].to(self.device)
                    gt = batch['gt'].to(self.device)
                    pred = self.model(x, idx)
                    loss = self.loss_fn(pred, gt)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, self.cfg['train']['grad_clip'])
                    optimizer.step()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                scheduler.step()

            save_checkpoint(
                self.model, optimizer, epochs, idx,
                os.path.join(self.output_dir, f'task_{idx}_final.pth')
            )
        self.evaluate_all()