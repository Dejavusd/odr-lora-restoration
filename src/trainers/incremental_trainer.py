import os
import time
import copy
import numpy as np
import torch
from tqdm import tqdm

from src.models.restoration_net import RestorationNet
from src.data.datasets import build_task_loaders
from src.losses.losses import (
    TotalLoss,
    compute_orthogonal_loss,
    compute_frequency_loss,
    compute_share_loss,
    compute_frequency_selection_loss,
)
from src.models.freq_decomp import (
    multilevel_dwt,
    freq_band_energy_from_bands,
    num_subbands,
)
from src.utils.metrics import compute_psnr, compute_ssim, LPIPSMetric
from src.utils.logger import get_logger, MetricLogger
from src.utils.checkpoint import save_checkpoint, load_checkpoint


class IncrementalTrainer:
    """
    增量训练器：按任务顺序训练，每个任务：
      1. 新增低秩分支
      2. 估计与旧任务相似度
      3. 校准 B_t，并记录当前任务频带能量
      4. 冻结基座和旧分支
      5. 只训练当前任务的 A_t / 共享 A_share
      6. 更新 DualGPM，并把当前任务频带能量保存为 old
      7. 保存 checkpoint
    全部任务训练完后合并分支，保存 merged.pth。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg['experiment'].get('device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
        self.output_dir = cfg['experiment']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = get_logger('trainer', self.output_dir)
        self.metric_logger = MetricLogger(os.path.join(self.output_dir, 'metrics.json'))

        self.task_order = [t['name'] for t in cfg['data']['tasks']]
        self.num_tasks = len(self.task_order)

        self.model = RestorationNet(cfg).to(self.device)
        self.loss_fn = TotalLoss(cfg).to(self.device)

        # 频带级别与子带数
        self.freq_level = cfg['odr_lora'].get('freq_level', 2)
        self.freq_bands = num_subbands(level=self.freq_level)

        # 记录每个任务的频带能量分布，用于估计任务相似度
        self.task_freq_energy = {}
        # 记录每个任务与旧任务的最大相似度
        self.task_similarity = {}

    # ------------------------------------------------------------------
    # 辅助：估计当前任务与旧任务的相似度（基于 DWT 子带能量）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _estimate_task_similarity(self, task_idx, train_loader):
        """
        通过 DWT 子带能量分布估计当前任务与之前任务的最大相似度。
        返回 [0, 1] 之间的标量。
        """
        if task_idx == 0:
            return 0.0

        energies = []
        self.model.eval()

        for i, batch in enumerate(train_loader):
            x = batch['input'].to(self.device)
            bands = multilevel_dwt(x, level=self.freq_level)
            ratio = freq_band_energy_from_bands(bands)
            energies.append(ratio)
            if i >= 4:
                break

        if len(energies) == 0:
            return 0.0

        avg_ratio = np.mean(energies, axis=0).tolist()
        self.task_freq_energy[self.task_order[task_idx]] = avg_ratio

        # 与之前所有任务比较，取最大余弦相似度
        sims = []
        for prev_idx in range(task_idx):
            prev_name = self.task_order[prev_idx]
            if prev_name in self.task_freq_energy:
                prev_ratio = self.task_freq_energy[prev_name]
                a = np.array(avg_ratio)
                b = np.array(prev_ratio)
                min_len = min(len(a), len(b))
                a = a[:min_len]
                b = b[:min_len]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                sims.append(cos)

        if len(sims) == 0:
            return 0.0
        return float(max(sims))

    # ------------------------------------------------------------------
    # 辅助：把当前任务的频带能量写入模型模块
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _record_current_freq_energy(self, train_loader):
        """
        遍历一批数据，计算 DWT 子带能量占比，写入所有 ODRLoRAConv2d 模块的
        freq_energy_current 属性。
        """
        self.model.eval()
        ratios = []
        for i, batch in enumerate(train_loader):
            x = batch['input'].to(self.device)
            bands = multilevel_dwt(x, level=self.freq_level)
            ratios.append(freq_band_energy_from_bands(bands))
            if i >= 4:
                break
        if len(ratios) == 0:
            return
        avg_ratio = np.mean(ratios, axis=0).tolist()
        for m in self.model.modules():
            if hasattr(m, 'task_B') and hasattr(m, 'dual_gpms'):
                m.freq_energy_current = torch.tensor(
                    avg_ratio, dtype=torch.float32, device=self.device
                )

    @torch.no_grad()
    def _promote_freq_energy_to_old(self):
        """
        训练完当前任务后，把 freq_energy_current 保存为 freq_energy_old，
        供下一个任务计算频带互补损失。
        """
        for m in self.model.modules():
            if hasattr(m, 'freq_energy_current'):
                m.freq_energy_old = m.freq_energy_current

    # ------------------------------------------------------------------
    # 训练单个任务
    # ------------------------------------------------------------------
    def train_task(self, task_name, task_idx, epochs):
        self.logger.info(f"===== 开始训练任务 {task_idx}: {task_name} =====")
        train_loader, val_loader = build_task_loaders(self.cfg, task_name)

        # 1. 新增任务分支
        self.model.add_task()
        self.model.set_current_task(task_idx)

        # 2. 估计任务相似度
        sim = self._estimate_task_similarity(task_idx, train_loader)
        self.task_similarity[task_idx] = sim
        self.logger.info(f"任务 {task_idx} 与旧任务最大相似度: {sim:.4f}")

        # 3. 校准 B_t
        self.logger.info(f"任务 {task_idx} 校准低秩基 B_t ...")
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                x = batch['input'].to(self.device)
                self.model.calibrate(x, task_idx, sim)
                if i >= 5:
                    break

        # 4. 记录当前任务频带能量
        self._record_current_freq_energy(train_loader)

        # 5. 冻结基座和旧分支，只训练当前任务相关参数
        self.model.freeze_base_and_old(task_idx)

        # 6. 收集可训练参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            self.logger.warning("没有可训练参数，跳过训练。")
            return

        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg['train']['lr'],
            weight_decay=self.cfg['train']['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # 7. 训练循环
        best_psnr = 0.0
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Task {task_idx} Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0

            for batch in pbar:
                x = batch['input'].to(self.device)
                gt = batch['gt'].to(self.device)
                pred = self.model(x, task_idx)
                loss = self.loss_fn(pred, gt)

                # 辅助损失
                orth_loss = compute_orthogonal_loss(self.model)
                share_loss = compute_share_loss(self.model)
                freq_loss = compute_frequency_loss(
                    self.model, task_idx, freq_bands=self.freq_bands
                )
                freq_sel_loss = compute_frequency_selection_loss(self.model)

                loss = loss \
                    + self.cfg['train']['loss']['orth'] * orth_loss \
                    + self.cfg['train']['loss']['freq'] * freq_loss \
                    + self.cfg['train']['loss']['share'] * share_loss \
                    + self.cfg['train']['loss'].get('freq_sel', 0.0) * freq_sel_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg['train']['grad_clip'])
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Task {task_idx} Epoch {epoch+1}: loss={avg_loss:.4f}")

            # 定期评估
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                metrics = self.evaluate_task(val_loader, task_idx)
                self.logger.info(
                    f"Task {task_idx} Epoch {epoch+1} 验证: "
                    f"PSNR={metrics['psnr']:.2f}, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"LPIPS={metrics['lpips']:.4f}"
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

        # 8. 训练结束后更新 DualGPM
        self.logger.info(f"任务 {task_idx} 更新 DualGPM ...")
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                x = batch['input'].to(self.device)
                self.model.update_gpm(x, task_idx)
                if i >= 5:
                    break

        # 9. 把 current 频带能量提升为 old，供下一个任务使用
        self._promote_freq_energy_to_old()

        # 10. 保存该任务最终 checkpoint
        save_checkpoint(
            self.model, optimizer, epochs, task_idx,
            os.path.join(self.output_dir, f'task_{task_idx}_final.pth')
        )
        self.logger.info(f"任务 {task_idx} 完成。")

    # ------------------------------------------------------------------
    # 评估单个任务
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_task(self, loader, task_idx):
        self.model.eval()
        psnrs, ssims, lpipses = [], [], []
        lpips_fn = LPIPSMetric(self.device)

        for batch in loader:
            x = batch['input'].to(self.device)
            gt = batch['gt'].to(self.device)
            pred = self.model(x, task_idx)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            lpipses.append(lpips_fn(pred, gt))

        return {
            'psnr': float(np.mean(psnrs)),
            'ssim': float(np.mean(ssims)),
            'lpips': float(np.mean(lpipses)),
        }

    # ------------------------------------------------------------------
    # 评估所有任务
    # ------------------------------------------------------------------
    def evaluate_all(self):
        results = {}
        for idx, task_name in enumerate(self.task_order):
            _, val_loader = build_task_loaders(self.cfg, task_name)
            metrics = self.evaluate_task(val_loader, idx)
            results[task_name] = metrics
            self.logger.info(f"任务 {task_name} 最终评估: {metrics}")

        avg_psnr = np.mean([r['psnr'] for r in results.values()])
        avg_ssim = np.mean([r['ssim'] for r in results.values()])
        self.logger.info(f"所有任务平均: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return results

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        epochs = self.cfg['train']['epochs_per_task']

        for idx, task_name in enumerate(self.task_order):
            self.train_task(task_name, idx, epochs)

        # 先评估未合并的模型（按任务 ID）
        self.logger.info("最终评估（按任务 ID，未合并）...")
        self.evaluate_all()

        # 合并所有任务分支
        self.logger.info("合并所有任务分支 ...")
        for idx in range(len(self.task_order)):
            self.model.merge_task(idx)

        save_checkpoint(
            self.model, None, epochs, len(self.task_order) - 1,
            os.path.join(self.output_dir, 'merged.pth')
        )
        self.metric_logger.save()
        self.logger.info("全部训练完成。")