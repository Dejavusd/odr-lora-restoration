import os
import time
import copy
import json
import numpy as np
import torch
from tqdm import tqdm

from src.models.restoration_net import RestorationNet
from src.data.datasets import build_task_loaders
from src.losses.losses import TotalLoss
from src.models.freq_decomp import (
    multilevel_dwt,
    freq_band_energy_from_bands,
    num_subbands,
)
from src.utils.metrics import compute_psnr, compute_ssim, LPIPSMetric
from src.utils.logger import get_logger, MetricLogger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.inference import predict_image


class IncrementalTrainer:
    """
    增量训练器：按任务顺序训练，每个任务：
      1. 新增低秩分支
      2. 估计与旧任务相似度
      3. 校准 B_t，并记录当前任务频带能量
      4. 冻结基座和旧分支
      5. 只训练当前任务的 A_t / 共享 A_share
      6. 更新 DualGPM，并把当前任务频带能量保存为 old
      7. 把当前分支合并进累积基座并保存 checkpoint

    每个任务结束后立即合并，因此新任务总是在上一任务的累积模型
    上学习；最终模型不需要任务 ID。
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
        self.continual_metrics_path = os.path.join(
            self.output_dir, 'continual_metrics.json'
        )
        self.performance_matrix = {}
        self.resume_state = None
        if os.path.isfile(self.continual_metrics_path):
            try:
                with open(self.continual_metrics_path, 'r') as handle:
                    self.performance_matrix = json.load(handle).get('matrix', {})
            except (OSError, ValueError):
                self.performance_matrix = {}

    # ------------------------------------------------------------------
    # 辅助：估计当前任务与旧任务的相似度（基于 DWT 子带能量）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _estimate_task_similarity(self, task_idx, train_loader):
        """
        通过 DWT 子带能量分布估计当前任务与之前任务的最大相似度。
        返回 [0, 1] 之间的标量。
        """
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

        if task_idx == 0:
            return 0.0

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

    def checkpoint_metadata(self):
        return {
            'task_freq_energy': self.task_freq_energy,
            'task_similarity': self.task_similarity,
            'performance_matrix': self.performance_matrix,
        }

    def restore_metadata(self, state):
        self.task_freq_energy = state.get('task_freq_energy', {})
        self.task_similarity = state.get('task_similarity', {})
        self.performance_matrix = state.get(
            'performance_matrix', self.performance_matrix
        )

    # ------------------------------------------------------------------
    # 训练单个任务
    # ------------------------------------------------------------------
    def train_task(self, task_name, task_idx, epochs, resume_state=None):
        self.logger.info(f"===== 开始训练任务 {task_idx}: {task_name} =====")
        train_loader, val_loader = build_task_loaders(self.cfg, task_name)

        is_resume = resume_state is not None
        start_epoch = int(resume_state.get('epoch', 0)) if is_resume else 0
        if is_resume:
            self.logger.info(f"从任务 {task_idx} 的第 {start_epoch + 1} 个 epoch 继续训练")
        else:
            # 1. 新增任务分支
            self.model.add_task()
            self.model.set_current_task(task_idx)

        # 2. 估计任务相似度
        if is_resume:
            sim = float(self.task_similarity.get(task_idx, 0.0))
        else:
            sim = self._estimate_task_similarity(task_idx, train_loader)
            self.task_similarity[task_idx] = sim
        self.logger.info(f"任务 {task_idx} 与旧任务最大相似度: {sim:.4f}")

        # 3. For later tasks, calibrate B_t on the cumulative backbone before
        # optimization. Task 0 first learns the base network; calibrating deep
        # features of a random backbone would create a meaningless shared basis.
        if task_idx > 0 and not is_resume:
            self.logger.info(f"任务 {task_idx} 校准低秩基 B_t ...")
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    x = batch['input'].to(self.device)
                    self.model.calibrate(x, task_idx, sim)
                    if i >= 5:
                        break

        # 4. 记录当前任务频带能量
        if not is_resume:
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
        if is_resume and resume_state.get('optimizer') is not None:
            optimizer.load_state_dict(resume_state['optimizer'])
            scheduler.step(start_epoch)

        # 7. 训练循环
        best_psnr = 0.0
        for epoch in range(start_epoch, epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Task {task_idx} Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0

            for batch in pbar:
                x = batch['input'].to(self.device)
                gt = batch['gt'].to(self.device)
                pred = self.model(x, task_idx)
                loss = self.loss_fn(pred, gt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg['train']['grad_clip'])
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Task {task_idx} Epoch {epoch+1}: loss={avg_loss:.4f}")

            save_checkpoint(
                self.model, optimizer, epoch + 1, task_idx,
                os.path.join(self.output_dir, 'latest.pth'),
                extra={**self.checkpoint_metadata(), 'in_progress': True},
            )

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
                        os.path.join(self.output_dir, f'task_{task_idx}_best.pth'),
                        extra=self.checkpoint_metadata(),
                    )

        # Initialize task-0 shared bases from the trained backbone.
        if task_idx == 0:
            self.logger.info("用已训练的首任务基座初始化共享子空间 ...")
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    x = batch['input'].to(self.device)
                    self.model.calibrate(x, task_idx, sim)
                    if i >= 5:
                        break

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

        # 10. Merge immediately. This makes the next task start from the
        # cumulative model and makes task-agnostic inference well-defined.
        self.model.merge_task(task_idx)
        self.model.set_current_task(-1)

        # Evaluate all seen tasks with the same cumulative model. This records
        # the R_{i,j} matrix required for ACC/BWT/forgetting.
        self._evaluate_seen_tasks(task_idx)

        # 11. 保存该任务最终 checkpoint
        save_checkpoint(
            self.model, optimizer, epochs, task_idx,
            os.path.join(self.output_dir, f'task_{task_idx}_final.pth'),
            extra=self.checkpoint_metadata(),
        )
        self.logger.info(f"任务 {task_idx} 完成。")

    # ------------------------------------------------------------------
    # 评估单个任务
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_task(self, loader, task_idx):
        self.model.eval()
        psnrs, ssims, lpipses = [], [], []
        groups = {}
        lpips_fn = LPIPSMetric(self.device)

        for batch in loader:
            x = batch['input'].to(self.device)
            gt = batch['gt'].to(self.device)
            eval_cfg = self.cfg.get('eval', {})
            pred = predict_image(self.model, x, task_idx,
                                 eval_cfg.get('tile_size', 0),
                                 eval_cfg.get('tile_overlap', 64))
            if bool(batch.get('grayscale', [False])[0]):
                pred = pred.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            lpipses.append(lpips_fn(pred, gt))
            if 'dataset_name' in batch:
                name = batch['dataset_name'][0]
                groups.setdefault(name, []).append((psnrs[-1], ssims[-1], lpipses[-1]))

        result = {
            'psnr': float(np.mean(psnrs)),
            'ssim': float(np.mean(ssims)),
            'lpips': float(np.mean(lpipses)),
        }
        if groups:
            result['datasets'] = {
                name: dict(zip(('psnr', 'ssim', 'lpips'),
                               map(float, np.mean(values, axis=0))), num=len(values))
                for name, values in groups.items()
            }
            self.logger.info(f"去噪分数据集评估: {result['datasets']}")
        return result

    # ------------------------------------------------------------------
    # 评估所有任务
    # ------------------------------------------------------------------
    def evaluate_all(self):
        results = {}
        for idx, task_name in enumerate(self.task_order):
            _, val_loader = build_task_loaders(self.cfg, task_name)
            # All old tasks are evaluated with the latest cumulative model.
            metrics = self.evaluate_task(val_loader, -1)
            results[task_name] = metrics
            self.logger.info(f"任务 {task_name} 最终评估: {metrics}")

        avg_psnr = np.mean([r['psnr'] for r in results.values()])
        avg_ssim = np.mean([r['ssim'] for r in results.values()])
        self.logger.info(f"所有任务平均: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        return results

    def _evaluate_seen_tasks(self, stage_idx):
        row = {}
        for eval_idx in range(stage_idx + 1):
            task_name = self.task_order[eval_idx]
            _, loader = build_task_loaders(self.cfg, task_name)
            row[task_name] = self.evaluate_task(loader, -1)
        self.performance_matrix[str(stage_idx)] = row

        psnr_row = [metrics['psnr'] for metrics in row.values()]
        acc = float(np.mean(psnr_row)) if psnr_row else 0.0
        backward = []
        forgetting = []
        for old_idx in range(stage_idx):
            name = self.task_order[old_idx]
            diagonal = self.performance_matrix.get(str(old_idx), {}).get(name)
            current = row.get(name)
            if diagonal is None or current is None:
                continue
            backward.append(current['psnr'] - diagonal['psnr'])
            historical = []
            for previous_stage in range(old_idx, stage_idx):
                value = self.performance_matrix.get(
                    str(previous_stage), {}
                ).get(name)
                if value is not None:
                    historical.append(value['psnr'])
            if historical:
                forgetting.append(max(historical) - current['psnr'])

        summary = {
            'stage': stage_idx,
            'acc_psnr': acc,
            'bwt_psnr': float(np.mean(backward)) if backward else 0.0,
            'forgetting_psnr': (
                float(np.mean(forgetting)) if forgetting else 0.0
            ),
        }
        with open(self.continual_metrics_path, 'w') as handle:
            json.dump(
                {'matrix': self.performance_matrix, 'latest': summary},
                handle, indent=2,
            )
        self.logger.info(f"持续学习指标: {summary}")

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, start_task_idx=0):
        epochs = self.cfg['train']['epochs_per_task']

        for idx, task_name in enumerate(self.task_order):
            if idx < start_task_idx:
                continue
            resume_state = None
            if self.resume_state is not None:
                if (int(self.resume_state.get('task_idx', -1)) == idx
                        and self.resume_state.get('in_progress', False)):
                    resume_state = self.resume_state
                self.resume_state = None
            self.train_task(task_name, idx, epochs, resume_state=resume_state)

        # 先评估未合并的模型（按任务 ID）
        self.logger.info("最终评估（按任务 ID，未合并）...")
        self.evaluate_all()

        # Each task has already been merged exactly once.
        self.model.prepare_for_inference()
        save_checkpoint(
            self.model, None, epochs, len(self.task_order) - 1,
            os.path.join(self.output_dir, 'merged.pth'),
            extra=self.checkpoint_metadata(),
        )
        self.metric_logger.save()
        self.logger.info("全部训练完成。")
