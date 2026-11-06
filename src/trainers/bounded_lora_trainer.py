"""Replay-free orthogonal LoRA with a per-layer Frobenius update bound."""
import copy
import math
from pathlib import Path

import numpy as np
import torch

from src.data.clean_protocol import sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.training_state import atomic_json


def bounded_config(source, output, max_relative_update=.05, epochs=20, batch_size=64,
                   smoke_test=False):
    if not math.isfinite(max_relative_update) or max_relative_update < 0:
        raise ValueError('max_relative_update 必须为有限非负数；0表示关闭上限')
    if min(epochs, batch_size) < 1:
        raise ValueError('epochs 和 batch_size 必须为正')
    state = torch.load(source, map_location='cpu', weights_only=False)
    if (state.get('clean_protocol_version') != 1 or state.get('task_idx') != 1
            or state.get('in_progress') or not state.get('gpm_updated')
            or state.get('smoke_test') or state.get('probe_only') or state.get('inference_only')
            or 'stability' in state.get('run_protocol', {})
            or 'update_bound' in state.get('run_protocol', {})):
        raise ValueError('共同起点必须是无回放 clean RainH task_1_final.pth')
    cfg = copy.deepcopy(state['run_protocol'])
    if not cfg['odr_lora'].get('full_update_protection'):
        raise ValueError('本实验要求完整更新正交保护已启用')
    cfg['experiment'].update(output_dir=str(output), memory_fraction=.9)
    cfg['train']['epochs_per_task'] = epochs
    cfg['eval']['validation_interval'] = 5
    for task in cfg['data']['tasks']:
        task['batch_size'] = batch_size
    cfg['update_bound'] = dict(source=str(Path(source).resolve()), source_sha256=sha256_file(source),
        max_relative_update=max_relative_update, norm='frobenius', denominator_epsilon=1e-12,
        selection='new_task_validation_psnr', replay=False, teacher=False,
        projection='scale_current_A_after_each_optimizer_step', optimizer_moments='unchanged')
    cfg['smoke_test'] = smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1, max_steps_per_epoch=2,
                            calibration_batches=1, gpm_batches=1)
        cfg['eval'].update(validation_interval=1, max_images=1)
        cfg['data']['num_workers'] = 0
        for task in cfg['data']['tasks']:
            task['batch_size'] = min(batch_size, 2)
    return cfg


@torch.no_grad()
def bound_updates(model, task_idx, max_ratio, epsilon=1e-12, enforce=True):
    """Scale A, preserving B and its GPM nullspace; W is frozen task-entry W."""
    if not math.isfinite(max_ratio) or max_ratio < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid update bound')
    layers = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, ODRLoRAConv2d):
            continue
        a, b = layer.task_A[task_idx], layer.task_B[task_idx]
        base_norm = layer.weight.float().norm().clamp_min(epsilon)
        update_norm = (a.float() @ b.float()).norm()
        ratio = float(update_norm/base_norm)
        if not math.isfinite(ratio):
            raise FloatingPointError(f'Nonfinite LoRA update: {name}')
        scale = 1.
        if enforce and max_ratio > 0 and ratio > max_ratio:
            scale = max_ratio/ratio * (1-1e-6)
            a.mul_(scale)
        after = float((a.float()@b.float()).norm()/base_norm)
        if enforce and max_ratio > 0 and after > max_ratio*(1+1e-5):
            raise RuntimeError(f'更新范数上限检查失败：{name} {after}')
        layers[name] = dict(before_ratio=ratio, after_ratio=after, scale=scale)
    if not layers:
        raise ValueError('No LoRA layers')
    return dict(max_before=max(v['before_ratio'] for v in layers.values()),
        max_after=max(v['after_ratio'] for v in layers.values()),
        clipped_layers=sum(v['scale']<1 for v in layers.values()), layers=layers)


class BoundedLoRATrainer(CleanSequenceTrainer):
    def __init__(self, cfg, manifest):
        super().__init__(cfg, manifest)
        self.bound_steps = []

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('bounded_lora_version') == 1:
            return super().restore(path)
        if sha256_file(path) != self.cfg['update_bound']['source_sha256']:
            raise ValueError('初始分支必须来自指定的共同 RainH 源断点')
        if state['split_sha256'] != self.split_sha:
            raise ValueError('源断点与当前数据划分 SHA256 不一致')
        _, _, state = load_checkpoint(self.model, str(path), map_location=self.device, return_state=True)
        self.model.load_state_dict(state['model'], strict=True)
        self.restore_metadata(state)
        self.history = []
        return state

    def payload(self, *args, **kwargs):
        result = super().payload(*args, **kwargs)
        result['bounded_lora_version'] = 1
        return result

    def after_optimizer_step(self, task_idx):
        if task_idx != 2:
            raise ValueError('此入口仅用于 Haze4K 对照')
        settings = self.cfg['update_bound']
        stats = bound_updates(self.model, task_idx, settings['max_relative_update'],
                              settings['denominator_epsilon'])
        self.bound_steps.append({k:v for k,v in stats.items() if k!='layers'})

    def training_diagnostics(self):
        result = dict(update_bound=dict(
            cap=self.cfg['update_bound']['max_relative_update'],
            steps=len(self.bound_steps),
            max_before=max(s['max_before'] for s in self.bound_steps),
            max_after=max(s['max_after'] for s in self.bound_steps),
            clipped_layer_steps=sum(s['clipped_layers'] for s in self.bound_steps)))
        self.bound_steps = []
        self.logger.info(f'UPDATE_BOUND: {result["update_bound"]}')
        return result

    def protection(self, idx):
        result = super().protection(idx)
        settings = self.cfg['update_bound']
        stats = bound_updates(self.model, idx, settings['max_relative_update'],
                              settings['denominator_epsilon'], enforce=False)
        if settings['max_relative_update'] > 0 and stats['max_after'] > settings['max_relative_update']*(1+1e-5):
            raise RuntimeError('验证最优模型超出更新范数上限')
        atomic_json(Path(self.output_dir)/f'task_{idx}_update_bound.json', dict(
            cap=settings['max_relative_update'], measured_before_merge=True, **stats))
        return result

    @torch.no_grad()
    def validation_for_selection(self, loader, task_idx):
        # Old validation images are forward-only diagnostics, never selection/loss.
        self.model.eval()
        metrics = {}
        for name in self.task_order[:task_idx+1]:
            val = loader if name == self.task_order[task_idx] else self.loader(name,'validation')
            rows = []
            for i, batch in enumerate(val):
                limit = self.cfg['eval'].get('max_images')
                if limit is not None and i >= limit:
                    break
                x, gt = batch['input'].to(self.device), batch['gt'].to(self.device)
                pred = self.model(x,task_idx)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('验证输出非有限')
                rows.append(dict(psnr=compute_psnr(pred,gt),ssim=compute_ssim(pred,gt),
                    lpips=float(self.loss_fn.perceptual.lpips(pred*2-1,gt*2-1).mean()),
                    raw_abs_max=float(pred.abs().max()),
                    out_of_range_fraction=float(((pred<0)|(pred>1)).float().mean())))
            if not rows:
                raise ValueError('空验证集')
            metrics[name] = {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            metrics[name].update(num=len(rows),worst_raw_abs_max=max(r['raw_abs_max'] for r in rows))
        return dict(psnr=metrics[self.task_order[task_idx]]['psnr'],
            selection='new_task_validation_psnr', tasks=metrics)
