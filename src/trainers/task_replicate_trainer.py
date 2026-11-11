"""Paired Haze4K seeds from one frozen RainH model and per-task summary bank."""
import copy
import math
from pathlib import Path

import torch

from src.data.clean_protocol import sha256_file
from src.models.odr_lora import ODRLoRAConv2d
from src.models.task_memory import bank_hash, select_task_memory
from src.trainers.memory_source_trainer import non_memory_fingerprint, non_memory_state_fingerprint
from src.trainers.task_balanced_trainer import TaskBalancedTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.training_state import atomic_json


def validate_rainh_source(state):
    if (state.get('task_balanced_version') != 1 or state.get('clean_protocol_version') != 1
            or state.get('task_idx') != 1 or state.get('in_progress') or not state.get('gpm_updated')
            or state.get('smoke_test') or state.get('inference_only') or state.get('probe_only')
            or state.get('task_replicate_version')):
        raise ValueError('Source must be a formal original task-memory RainH task_1_final.pth')
    cfg = state['run_protocol']
    if (cfg['odr_lora'].get('gpm_update_rule') != 'task_bank'
            or cfg['odr_lora'].get('gpm_statistics') != 'raw'
            or cfg['odr_lora'].get('basis_construction') != 'projected_svd'
            or not cfg['odr_lora'].get('full_update_protection')
            or cfg['odr_lora'].get('skip_protection') != 'fused'
            or cfg['task_memory'].get('branch_B') or cfg['task_memory'].get('replay')):
        raise ValueError('Unsupported source protection protocol')
    bank = state.get('task_sketches')
    if not isinstance(bank, dict) or bank_hash(bank) != state.get('task_sketches_sha256'):
        raise ValueError('Missing or corrupted source task-summary bank')
    suffix = '.dual_gpms.global._basis'
    dimensions = {k[:-len(suffix)]: v.shape[0] for k, v in state['model'].items() if k.endswith(suffix)}
    if not dimensions or set(bank) != set(dimensions):
        raise ValueError('Source summary layers do not match model')
    for name, entries in bank.items():
        if set(entries) != {'0', '1'}:
            raise ValueError('Source must retain both LSRW and RainH summaries only')
        for entry in entries.values():
            F = entry['factor']
            if (F.ndim != 2 or F.shape[0] != dimensions[name]
                    or F.shape[1] > min(dimensions[name], cfg['task_memory']['sketch_rank'])
                    or not torch.isfinite(F).all() or entry['columns'] < 1
                    or not math.isfinite(entry['total_energy']) or entry['total_energy'] < 0):
                raise ValueError('Invalid source task summary')
    if set(state['performance_matrix']) != {'0', '1'}:
        raise ValueError('Source must contain exactly the first two evaluation stages')


def task_replicate_config(source, output, mode='balanced', seed=7, epochs=20, batch_size=64, smoke_test=False):
    if mode not in ('balanced', 'mean_pca') or min(epochs, batch_size) < 1:
        raise ValueError('Invalid mode or training budget')
    if type(seed) is not int or not 0 <= seed < 2**32 - 20000:
        raise ValueError('Seed must support the existing task-index offset of 20000')
    state = torch.load(source, map_location='cpu', weights_only=False)
    validate_rainh_source(state)
    cfg = copy.deepcopy(state['run_protocol'])
    cfg['experiment'].update(output_dir=str(output), incremental_seed=seed, memory_fraction=.9)
    cfg['train'].pop('epochs_by_task', None)
    cfg['train']['epochs_per_task'] = epochs
    for task in cfg['data']['tasks']:
        task['batch_size'] = batch_size
    cfg['task_memory']['mode'] = mode
    cfg['task_memory']['selector'] = ('worst normalized sketch coverage first' if mode == 'balanced'
                                      else 'equal-task mean covariance PCA')
    cfg['task_replicate'] = dict(version=1, source=str(Path(source).resolve()),
        source_sha256=sha256_file(source), source_split_sha256=state['split_sha256'],
        source_physical_sha256=non_memory_state_fingerprint(state['model']),
        source_bank_sha256=bank_hash(state['task_sketches']), seed=seed,
        effective_haze_seed=seed+20000,
        boundary_seed=cfg['experiment']['seed']+900002,
        scope='conditional on fixed LSRW/RainH weights and snapshots; vary Haze4K training seed only',
        replay=False, rank_change=False, current_task_only='dehaze')
    cfg['smoke_test'] = smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1, max_steps_per_epoch=1, calibration_batches=1, gpm_batches=1)
        cfg['eval'].update(validation_interval=1, max_images=1)
        cfg['data']['num_workers'] = 0
        for task in cfg['data']['tasks']:
            task['batch_size'] = min(batch_size, 2)
    return cfg


class TaskReplicateTrainer(TaskBalancedTrainer):
    def payload(self, *args, **kwargs):
        state = super().payload(*args, **kwargs)
        state['task_replicate_version'] = 1
        return state

    @torch.no_grad()
    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('task_replicate_version') == 1:
            return super().restore(path)
        settings = self.cfg['task_replicate']
        if sha256_file(path) != settings['source_sha256']:
            raise ValueError('Must branch from the declared common RainH source')
        validate_rainh_source(state)
        if state['split_sha256'] != self.split_sha:
            raise ValueError('Source split SHA256 mismatch')
        load_checkpoint(self.model, str(path), map_location=self.device)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.set_current_task(-1)
        self.task_sketches = state['task_sketches']
        self.restore_metadata(state)
        before = non_memory_fingerprint(self.model)
        if before != settings['source_physical_sha256']:
            raise ValueError('Source physical weights changed while loading')
        sample = torch.linspace(0, 1, 3*31*29, device=self.device).reshape(1, 3, 31, 29)
        self.model.eval()
        prediction = self.model(sample, -1)
        layers = {}
        for name, m in self.model.named_modules():
            if not isinstance(m, ODRLoRAConv2d):
                continue
            basis, info = select_task_memory(self.task_sketches[name], m.dual_gpms['global'].max_basis,
                                             self.cfg['task_memory']['mode'])
            m.dual_gpms['global'].basis = basis.to(device=m.weight.device, dtype=m.weight.dtype)
            m.dual_gpms['global'].singular_values = None
            layers[name] = info
        error = float((self.model(sample, -1)-prediction).abs().max())
        after = non_memory_fingerprint(self.model)
        if before != after or error != 0 or bank_hash(self.task_sketches) != settings['source_bank_sha256']:
            raise RuntimeError('Reselecting GPM changed common weights, snapshots or forward')
        self.history = []
        self.best_psnr = self.best_epoch = self.best_path = self.frozen_sha = None
        atomic_json(Path(self.output_dir)/'rainh_boundary_reselect.json', dict(
            source_sha256=settings['source_sha256'], physical_sha256=after,
            bank_sha256=bank_hash(self.task_sketches), mode=self.cfg['task_memory']['mode'],
            forward_max_error=error, layers=layers,
            note='Reselect GPM from frozen snapshots only; no TRAIN/validation/test images read'))
        extra = dict(in_progress=False, gpm_updated=True, memory_reselected=True,
                     selected_epoch=state.get('selected_epoch'))
        self.save(Path(self.output_dir)/'task_1_final.pth', 1, 0, **extra)
        self.save(Path(self.output_dir)/'latest.pth', 1, 0, **extra)
        return self.payload(1, 0, **extra)
