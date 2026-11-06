"""Opt-in clean benchmark: grouped validation, best restore, complete resume.

The legacy trainer is deliberately unchanged. GPM-32 still compresses memory;
record that transition instead of claiming that all old spans are retained.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.clean_protocol import make_loader, sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.incremental_trainer import IncrementalTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.inference import predict_image
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.training_state import atomic_checkpoint, atomic_json, capture_rng, restore_rng, seed_all


def run_protocol(cfg):
    result = copy.deepcopy(cfg)
    for key in ('output_dir', 'name'):
        result['experiment'].pop(key, None)
    return result


def validate_resume(cfg, saved, split_sha, allow_batch_size_change=False):
    if (saved.get('clean_protocol_version') != 1 or saved.get('probe_only')
            or saved.get('inference_only') or saved.get('smoke_test', False) != cfg.get('smoke_test', False)):
        raise ValueError('只接受本 clean 入口产生的同模式 checkpoint；旧模型、probe、merged.pth 不可续训')
    if saved.get('split_sha256') != split_sha:
        raise ValueError('恢复模型与当前数据划分 SHA256 不一致')
    old, new = copy.deepcopy(saved['run_protocol']), run_protocol(cfg)
    if allow_batch_size_change:
        if saved.get('in_progress', False):
            raise ValueError('batch size 只能在已完成任务的边界更改，不能用于中途精确恢复')
        for item in (old, new):
            # Batch64 + VGG can exceed the previous 50% allocator quota.
            item['experiment'].pop('memory_fraction', None)
            for task in item['data']['tasks']:
                task.pop('batch_size', None)
    if not saved.get('in_progress', False) and saved.get('task_idx') == 0:
        # One clean task-0 checkpoint is the common starting point for all arms.
        for item in (old, new):
            item['experiment'].pop('method', None)
            item['experiment'].pop('incremental_seed', None)
            item['model'].pop('new_task_private_rank_base', None)
            item['odr_lora'].pop('basis_construction', None)
    if old != new:
        raise ValueError('恢复时训练协议发生改变（中途不能更换方法、种子、预算或数据）')
    if not saved.get('in_progress', False) and not saved.get('gpm_updated', False):
        raise ValueError('任务末 GPM 尚未更新，不能开始下一任务')


def frozen_fingerprint(model, task_idx):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.endswith(f'.task_A.{task_idx}'):
            continue
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class CleanSequenceTrainer(IncrementalTrainer):
    def __init__(self, cfg, manifest, allow_batch_size_change=False):
        super().__init__(cfg)
        self.allow_batch_size_change = allow_batch_size_change
        self.manifest = manifest
        self.split_sha = sha256_file(cfg['data']['split_manifest'])
        self.protocol = run_protocol(cfg)
        self.history = []
        self.best_psnr = self.best_epoch = self.best_path = None
        self.frozen_sha = None
        self.loss_fn.eval().requires_grad_(False)
        if not self.loss_fn.perceptual.use_lpips:
            raise RuntimeError('clean 实验要求真实 LPIPS；禁止静默回退为 L1')

    def loader(self, task, role):
        return make_loader(self.cfg, self.manifest, task, role)

    def payload(self, task_idx, epoch, optimizer=None, scheduler=None, **extra):
        return dict(model=self.model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            scheduler=scheduler.state_dict() if scheduler else None, epoch=epoch, task_idx=task_idx,
            current_task_idx=task_idx if extra.get('in_progress', False) else -1,
            num_tasks=len(self.model.backbone.input_conv.task_A), rng=capture_rng(),
            clean_protocol_version=1, split_sha256=self.split_sha, run_protocol=self.protocol,
            smoke_test=self.cfg.get('smoke_test', False), best_psnr=self.best_psnr,
            best_epoch=self.best_epoch, best_path=self.best_path, history=self.history,
            frozen_sha=self.frozen_sha, **self.checkpoint_metadata(), **extra)

    def save(self, path, task_idx, epoch, optimizer=None, scheduler=None, **extra):
        atomic_checkpoint(path, self.payload(task_idx, epoch, optimizer, scheduler, **extra))

    def restore(self, path):
        _, _, state = load_checkpoint(self.model, str(path), map_location=self.device, return_state=True)
        validate_resume(self.cfg, state, self.split_sha, self.allow_batch_size_change)
        self.model.load_state_dict(state['model'], strict=True)
        self.restore_metadata(state)
        self.history = state.get('history', [])
        return state

    @torch.no_grad()
    def evaluate_task(self, loader, task_idx):
        self.model.eval()
        rows = []
        limit = self.cfg['eval'].get('max_images')
        for index, batch in enumerate(loader):
            if limit is not None and index >= limit:
                break
            x, gt = batch['input'].to(self.device), batch['gt'].to(self.device)
            pred = predict_image(self.model, x, task_idx, self.cfg['eval'].get('tile_size', 0),
                                 self.cfg['eval'].get('tile_overlap', 64))
            if not torch.isfinite(pred).all():
                raise FloatingPointError('评估输出包含 NaN/Inf')
            rows.append((compute_psnr(pred, gt), compute_ssim(pred, gt),
                         float(self.loss_fn.perceptual.lpips(pred*2-1, gt*2-1).mean())))
        if not rows:
            raise ValueError('评估集不能为空')
        result = dict(zip(('psnr', 'ssim', 'lpips'), map(float, np.mean(rows, axis=0))))
        result['num'] = len(rows)
        return result

    @torch.no_grad()
    def protection(self, idx):
        values = []
        for layer in self.model.modules():
            if isinstance(layer, ODRLoRAConv2d) and layer.dual_gpms['global'].basis is not None:
                delta = layer.task_A[idx] @ layer.task_B[idx]
                values.append(float((delta @ layer.dual_gpms['global'].basis).norm()) /
                              max(float(delta.norm()), 1e-12))
        if not values:
            raise ValueError('缺少旧 GPM，不能声称保护旧任务')
        if max(values) > 1e-5:
            raise RuntimeError(f'完整更新泄漏超标：{max(values)}')
        return dict(mean=float(np.mean(values)), max=max(values), layers=len(values))

    def backward_batch(self, x, gt, task_idx):
        loss = self.loss_fn(self.model(x, task_idx), gt)
        if not torch.isfinite(loss):
            raise FloatingPointError('训练 loss 出现 NaN/Inf')
        loss.backward()
        return float(loss.detach())

    def validation_for_selection(self, loader, task_idx):
        return self.evaluate_task(loader, task_idx)

    def after_optimizer_step(self, task_idx):
        pass

    def training_diagnostics(self):
        return {}

    def train_task(self, task_name, task_idx, epochs, resume_state=None):
        train, val = self.loader(task_name, 'train'), self.loader(task_name, 'validation')
        self.logger.info(f'CLEAN task={task_idx} {task_name}: train={len(train.dataset)}, '
                         f'validation={len(val.dataset)}, epochs={epochs}')
        start = 0
        if resume_state is None:
            seed = self.cfg['experiment']['seed'] if task_idx == 0 else self.cfg['experiment']['incremental_seed']
            seed_all(seed + task_idx*10000)
            self.model.add_task()
            self.model.set_current_task(task_idx)
            similarity = self._estimate_task_similarity(task_idx, train)
            self.task_similarity[task_idx] = similarity
            self._record_current_freq_energy(train)
            if task_idx:
                self._calibrate(train, task_idx, similarity)
            self.best_psnr = self.best_epoch = self.best_path = None
            self.frozen_sha = frozen_fingerprint(self.model, task_idx) if task_idx else None
        else:
            start = int(resume_state['epoch'])
            self.model.set_current_task(task_idx)
            self.best_psnr = resume_state['best_psnr']
            self.best_epoch = resume_state['best_epoch']
            self.best_path = resume_state['best_path']
            self.frozen_sha = resume_state['frozen_sha']
            if self.best_path and not Path(self.best_path).is_file():
                if self.best_epoch == start:
                    self.best_path = self.resume_path
                else:
                    raise FileNotFoundError(f'缺少验证最优 checkpoint：{self.best_path}')
            if task_idx and frozen_fingerprint(self.model, task_idx) != self.frozen_sha:
                raise ValueError('恢复时冻结张量的指纹不一致')
        self.model.freeze_base_and_old(task_idx)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg['train']['lr'],
                                     weight_decay=self.cfg['train']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                             eta_min=self.cfg['train'].get('eta_min', 1e-6))
        if resume_state is not None:
            if resume_state.get('optimizer') is None or resume_state.get('scheduler') is None:
                raise ValueError('中途恢复缺少优化器或 scheduler，拒绝不精确恢复')
            optimizer.load_state_dict(resume_state['optimizer'])
            scheduler.load_state_dict(resume_state['scheduler'])
            restore_rng(resume_state['rng'])
        for epoch in range(start, epochs):
            self.model.train()
            losses = []
            for step, batch in enumerate(tqdm(train, desc=f'CLEAN {task_name} {epoch+1}/{epochs}'), 1):
                x, gt = batch['input'].to(self.device), batch['gt'].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss_value = self.backward_batch(x, gt, task_idx)
                torch.nn.utils.clip_grad_norm_(params, self.cfg['train']['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                self.after_optimizer_step(task_idx)
                losses.append(loss_value)
                limit = self.cfg['train'].get('max_steps_per_epoch')
                if limit is not None and step >= limit:
                    break
            if not losses:
                raise ValueError('训练集不能为空')
            scheduler.step()
            row = dict(task=task_name, epoch=epoch+1, loss=float(np.mean(losses)),
                       lr=optimizer.param_groups[0]['lr'], optimizer_updates=len(losses))
            row.update(self.training_diagnostics())
            interval = self.cfg['eval']['validation_interval']
            improved = False
            if (epoch+1) % interval == 0 or epoch+1 == epochs:
                row['validation'] = self.validation_for_selection(val, task_idx)
                score = row['validation']['psnr']
                if self.best_psnr is None or score > self.best_psnr:
                    self.best_psnr, self.best_epoch = score, epoch+1
                    self.best_path = str((Path(self.output_dir)/f'task_{task_idx}_best.pth').resolve())
                    improved = True
                self.logger.info(f'VALID {task_name} epoch={epoch+1}: {row["validation"]}')
            self.history.append(row)
            if improved:
                self.save(self.best_path, task_idx, epoch+1, optimizer, scheduler,
                          in_progress=True, gpm_updated=False)
            self.save(Path(self.output_dir)/'latest.pth', task_idx, epoch+1, optimizer, scheduler,
                      in_progress=True, gpm_updated=False)
            atomic_json(Path(self.output_dir)/'history.json', self.history)
            self.logger.info(f'TRAIN {task_name} epoch={epoch+1}: loss={row["loss"]:.5f}')
        if not self.best_path:
            raise RuntimeError('未生成验证最优模型')
        # Restore ONLY selected weights: do not rewind history or the epoch budget.
        _, _, selected = load_checkpoint(self.model, self.best_path, map_location=self.device, return_state=True)
        self.model.load_state_dict(selected['model'], strict=True)
        del selected
        self.model.set_current_task(task_idx)
        if task_idx and frozen_fingerprint(self.model, task_idx) != self.frozen_sha:
            raise RuntimeError('训练修改了冻结的旧参数/B/GPM')
        if task_idx:
            atomic_json(Path(self.output_dir)/f'task_{task_idx}_protection.json', self.protection(task_idx))
        # Deterministic TRAIN-only postprocessing, independent of selected epoch RNG.
        seed_all(self.cfg['experiment']['seed'] + 900000 + task_idx)
        if task_idx == 0:
            # Initialize all arms' common task-0 bases with the SAME legacy rule.
            layers = [m for m in self.model.modules() if isinstance(m, ODRLoRAConv2d)]
            modes = [m.basis_construction for m in layers]
            try:
                for m in layers:
                    m.basis_construction = 'legacy'
                self._calibrate(train, task_idx, 0.)
            finally:
                for m, mode in zip(layers, modes):
                    m.basis_construction = mode
        self._finish_task(train, task_idx)
        self._evaluate_seen_tasks(task_idx)
        self.frozen_sha = None
        final = self.payload(task_idx, epochs, in_progress=False, gpm_updated=True,
                             selected_epoch=self.best_epoch, selection_metric='train_holdout_psnr')
        atomic_checkpoint(Path(self.output_dir)/f'task_{task_idx}_final.pth', final)
        atomic_checkpoint(Path(self.output_dir)/'latest.pth', final)
        self.logger.info(f'完成 {task_name}；使用验证最优 epoch={self.best_epoch} 更新记忆并合并')

    @torch.no_grad()
    def _calibrate(self, loader, idx, similarity):
        self.model.eval()
        for i, batch in enumerate(loader):
            self.model.calibrate(batch['input'].to(self.device), idx, similarity)
            if i+1 >= self.cfg['train']['calibration_batches']:
                break

    @torch.no_grad()
    def finalize_memory(self, idx):
        pass

    @torch.no_grad()
    def _finish_task(self, loader, idx):
        self.model.eval()
        sample = next(iter(loader))['input'][:1].to(self.device)
        # Fixed odd crop also exercises padding/cropping in the actual backbone.
        sample = sample[..., :min(127, sample.shape[-2]), :min(125, sample.shape[-1])]
        before = self.model(sample, idx)
        old = {name: m.dual_gpms['global'].basis.detach().clone()
               for name, m in self.model.named_modules()
               if isinstance(m, ODRLoRAConv2d) and m.dual_gpms['global'].basis is not None}
        for i, batch in enumerate(loader):
            self.model.update_gpm(batch['input'].to(self.device), idx)
            if i+1 >= self.cfg['train']['gpm_batches']:
                break
        self.finalize_memory(idx)
        memory_error = float((self.model(sample, idx)-before).abs().max())
        if memory_error != 0:
            raise RuntimeError(f'更新 GPM 改变了物理前向函数：{memory_error}')
        transition = {}
        for name, layer in self.model.named_modules():
            if not isinstance(layer, ODRLoRAConv2d):
                continue
            new = layer.dual_gpms['global'].basis
            if new is None:
                raise RuntimeError(f'任务末 GPM 为空：{name}')
            previous = old.get(name)
            overlap_loss = (float((previous-new@(new.T@previous)).norm()) /
                            max(float(previous.norm()), 1e-12)) if previous is not None else 0.
            transition[name] = dict(rank=new.shape[1], cap=layer.dual_gpms['global'].max_basis,
                                    old_span_residual=overlap_loss)
        self.model.merge_task(idx)
        self.model.set_current_task(-1)
        error = (self.model(sample, -1)-before).abs()
        if float(error.max()) > 1e-3 or float(error.mean()) > 1e-4:
            raise RuntimeError('任务末合并的像素误差超标')
        residual = [r['old_span_residual'] for r in transition.values()]
        atomic_json(Path(self.output_dir)/f'task_{idx}_gpm_transition.json', dict(
            updated_from='training partition only, AFTER restoring validation-best weights',
            gpm_forward_max_error=memory_error, merge_max_error=float(error.max()),
            merge_mean_error=float(error.mean()), old_span_residual_mean=float(np.mean(residual)),
            old_span_residual_max=max(residual), layers=transition))
        if max(residual) > 1e-5:
            self.logger.warning('GPM-32 压缩丢弃了部分旧方向；详见 gpm_transition，不能声称旧子空间全部保留')

    def _evaluate_seen_tasks(self, stage_idx):
        row = {name: self.evaluate_task(self.loader(name, 'test'), -1)
               for name in self.task_order[:stage_idx+1]}
        self.performance_matrix[str(stage_idx)] = row
        backward, forgetting = [], []
        for old_idx in range(stage_idx):
            name = self.task_order[old_idx]
            backward.append(row[name]['psnr']-self.performance_matrix[str(old_idx)][name]['psnr'])
            historical = [self.performance_matrix[str(j)][name]['psnr'] for j in range(old_idx, stage_idx)]
            forgetting.append(max(0., max(historical)-row[name]['psnr']))
        summary = dict(stage=stage_idx, acc_psnr=float(np.mean([v['psnr'] for v in row.values()])),
                       bwt_psnr=float(np.mean(backward)) if backward else 0.,
                       forgetting_psnr=float(np.mean(forgetting)) if forgetting else 0.)
        atomic_json(self.continual_metrics_path, dict(matrix=self.performance_matrix, latest=summary))
        self.logger.info(f'TEST stage={stage_idx}: {row} continual={summary}')

    def run(self, stop_after=None, resume_path=None):
        start, state = 0, None
        if resume_path:
            self.resume_path = str(Path(resume_path).resolve())
            state = self.restore(resume_path)
            start = state['task_idx'] + (0 if state.get('in_progress') else 1)
        stop = self.task_order.index(stop_after) if stop_after else len(self.task_order)-1
        for idx in range(start, stop+1):
            saved = state if state is not None and state.get('in_progress') and idx == start else None
            self.train_task(self.task_order[idx], idx, self.cfg['train']['epochs_per_task'], saved)
            state = None
        # Keep the live checkpoint and its GPM intact; export a separate copy.
        exported = copy.deepcopy(self.model)
        exported.prepare_for_inference()
        atomic_checkpoint(Path(self.output_dir)/'merged.pth', dict(model=exported.state_dict(),
            num_tasks=0, current_task_idx=-1, inference_only=True, clean_protocol_version=1,
            split_sha256=self.split_sha, run_protocol=self.protocol, smoke_test=self.cfg.get('smoke_test', False)))
        atomic_json(Path(self.output_dir)/'status.json', dict(status='complete', stopped_after=self.task_order[stop],
                    smoke_test=self.cfg.get('smoke_test', False)))
