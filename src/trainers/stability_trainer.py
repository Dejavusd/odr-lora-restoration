"""Opt-in, matched Haze4K control/replay experiment from a clean RainH final."""
import copy
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.data.datasets import RestorationDataset
from src.data.clean_protocol import sha256_file
from src.scripts.audit_output_stability import read_model
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.training_state import atomic_json


def stability_config(source, output, mode='replay_distill', epochs=20, batch_size=64,
                     memory_per_task=64, smoke_test=False):
    state = torch.load(source, map_location='cpu', weights_only=False)
    if (state.get('clean_protocol_version') != 1 or state.get('task_idx') != 1
            or state.get('in_progress') or not state.get('gpm_updated')
            or state.get('smoke_test') or state.get('probe_only') or state.get('inference_only')
            or state.get('stability_control_version')):
        raise ValueError('需要正式 clean RainH task_1_final.pth 作为共同起点')
    cfg = copy.deepcopy(state['run_protocol'])
    cfg['experiment'].update(output_dir=str(output), memory_fraction=.9)
    cfg['train']['epochs_per_task'] = epochs
    cfg['eval']['validation_interval'] = 5
    for task in cfg['data']['tasks']:
        task['batch_size'] = batch_size
    cfg['stability'] = dict(mode=mode, source=str(Path(source).resolve()),
        source_sha256=sha256_file(source), memory_per_task=memory_per_task,
        replay_l1_weight=1., distill_weight=1., range_weight=.1,
        replay_images_per_task_per_update=1, selection='macro_seen_validation_psnr',
        memory_seed=20261104, full_image_replay=True)
    cfg['smoke_test'] = smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1, max_steps_per_epoch=1,
                            calibration_batches=1, gpm_batches=1)
        cfg['eval'].update(validation_interval=1, max_images=1)
        cfg['data']['num_workers'] = 0
        for task in cfg['data']['tasks']:
            task['batch_size'] = min(batch_size, 2)
    return cfg


def range_loss(pred):
    return (F.relu(-pred) + F.relu(pred-1)).mean()


def replay_loss(pred, gt, teacher, settings):
    # Do not reproduce the teacher's known unbounded full-image outliers.
    target = teacher.detach().clamp(0, 1)
    return (settings['replay_l1_weight'] * F.l1_loss(pred, gt)
            + settings['distill_weight'] * F.l1_loss(pred, target)
            + settings['range_weight'] * range_loss(pred))


class StabilityTrainer(CleanSequenceTrainer):
    def __init__(self, cfg, manifest):
        super().__init__(cfg, manifest)
        settings = cfg['stability']
        if settings['mode'] not in ('control', 'replay_distill'):
            raise ValueError('Unknown stability mode')
        if sha256_file(settings['source']) != settings['source_sha256']:
            raise ValueError('教师源断点 SHA256 变化')
        self.teacher, source = read_model(settings['source'], self.device)
        if source['split_sha256'] != self.split_sha:
            raise ValueError('教师与当前数据划分不同')
        self.memories, memory_rows = {}, {}
        for task in self.task_order[:2]:
            item = manifest['tasks'][task]
            indices = list(item['train_indices'])
            random.Random(settings['memory_seed']).shuffle(indices)
            chosen, groups = [], set()
            for index in indices:
                row = item['train_records'][index]
                if row['gt_group'] not in groups:
                    chosen.append(index)
                    groups.add(row['gt_group'])
                if len(chosen) >= settings['memory_per_task']:
                    break
            rows = [item['train_records'][i] for i in chosen]
            memory_rows[task] = dict(train_indices=chosen, records=rows)
            # augment=False intentionally retains FULL resolution, including borders.
            self.memories[task] = RestorationDataset([r['input'] for r in rows],
                [r['gt'] for r in rows], augment=False, strict_size=True)
        path = Path(self.output_dir)/'replay_manifest.json'
        atomic_json(path, dict(partition='TRAIN only; one degraded input per unique GT',
                              full_resolution=True, tasks=memory_rows))
        self.loss_components = {}

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('stability_control_version') == 1:
            return super().restore(path)
        # This dedicated entry derives all architecture/GPM settings from this
        # exact formal source. Changed budget/loss/selection are explicit in cfg.
        if sha256_file(path) != self.cfg['stability']['source_sha256']:
            raise ValueError('初始分支必须来自指定的共同 RainH 源断点')
        _, _, state = load_checkpoint(self.model, str(path), map_location=self.device, return_state=True)
        self.model.load_state_dict(state['model'], strict=True)
        self.restore_metadata(state)
        self.history = []
        return state

    def payload(self, *args, **kwargs):
        result = super().payload(*args, **kwargs)
        result['stability_control_version'] = 1
        return result

    def backward_batch(self, x, gt, task_idx):
        if task_idx != 2:
            raise ValueError('该对照入口仅训练 Haze4K')
        settings = self.cfg['stability']
        pred = self.model(x, task_idx)
        loss = self.loss_fn(pred, gt)
        if settings['mode'] == 'replay_distill':
            loss = loss + settings['range_weight'] * range_loss(pred)
        if not torch.isfinite(loss):
            raise FloatingPointError('新任务 loss 非有限')
        total = float(loss.detach())
        # Release the batch64 graph before full-resolution replay to bound VRAM.
        loss.backward()
        del pred, loss
        self.loss_components = dict(new_task=total)
        if settings['mode'] == 'replay_distill':
            for name, dataset in self.memories.items():
                batch = dataset[random.randrange(len(dataset))]
                old_x = batch['input'][None].to(self.device)
                old_gt = batch['gt'][None].to(self.device)
                with torch.no_grad():
                    target = self.teacher(old_x, -1)
                old_pred = self.model(old_x, task_idx)
                if not torch.isfinite(target).all() or not torch.isfinite(old_pred).all():
                    raise FloatingPointError(f'{name} 回放输出非有限')
                old_loss = replay_loss(old_pred, old_gt, target, settings) / len(self.memories)
                if not torch.isfinite(old_loss):
                    raise FloatingPointError('回放 loss 非有限')
                total += float(old_loss.detach())
                self.loss_components[name] = float(old_loss.detach())
                old_loss.backward()
                del old_pred, old_loss, target
        return total

    @torch.no_grad()
    def validation_for_selection(self, loader, task_idx):
        self.model.eval()
        metrics = {}
        for name in self.task_order[:task_idx+1]:
            rows = []
            for index, batch in enumerate(self.loader(name, 'validation')):
                limit = self.cfg['eval'].get('max_images')
                if limit is not None and index >= limit:
                    break
                x, gt = batch['input'].to(self.device), batch['gt'].to(self.device)
                pred = self.model(x, task_idx)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('全图验证输出非有限')
                row = dict(psnr=compute_psnr(pred,gt), ssim=compute_ssim(pred,gt),
                    raw_abs_max=float(pred.abs().max()),
                    out_of_range_fraction=float(((pred<0)|(pred>1)).float().mean()))
                if name in self.memories:
                    reference = self.teacher(x, -1)
                    if not torch.isfinite(reference).all():
                        raise FloatingPointError('教师验证输出非有限')
                    row['clamped_teacher_mae'] = float((pred.clamp(0,1)-reference.clamp(0,1)).abs().mean())
                    row['raw_to_bounded_teacher_mae'] = float((pred-reference.clamp(0,1)).abs().mean())
                rows.append(row)
            if not rows:
                raise ValueError('空验证集')
            metrics[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            metrics[name]['worst_raw_abs_max'] = max(r['raw_abs_max'] for r in rows)
            metrics[name]['num'] = len(rows)
        result = dict(psnr=float(np.mean([m['psnr'] for m in metrics.values()])),
            selection='macro_seen_validation_psnr', tasks=metrics,
            last_step_losses=self.loss_components)
        self.logger.info(f'STABILITY validation: {result}')
        return result
