"""Append denoise/GOPRO without dropping frozen per-task activation summaries."""
import copy
import json
import math
from pathlib import Path

import torch

from src.data.clean_protocol import sha256_file
from src.data.five_task_protocol import load_extension, new_tasks
from src.models.task_memory import bank_hash
from src.trainers.five_task_trainer import FiveTaskTrainer
from src.trainers.memory_source_trainer import non_memory_fingerprint, non_memory_state_fingerprint
from src.trainers.task_balanced_trainer import TaskBalancedTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.training_state import atomic_json


OLD_TASKS = ('lowlight', 'rainh', 'dehaze')


def validate_task_memory_source(state):
    if (state.get('clean_protocol_version') != 1 or state.get('task_balanced_version') != 1
            or state.get('task_idx') != 2 or state.get('num_tasks') != 3
            or state.get('in_progress') or not state.get('gpm_updated')
            or any(state.get(k) for k in ('smoke_test', 'inference_only', 'probe_only',
                                         'task_memory_extension_version', 'five_task_version'))):
        raise ValueError('Source must be a formal three-task task-memory task_2_final.pth')
    cfg = state['run_protocol']
    memory = cfg['task_memory']
    odr = cfg['odr_lora']
    if ([t['name'] for t in cfg['data']['tasks']] != list(OLD_TASKS)
            or cfg['model'].get('architecture') != 'stable_v1'
            or memory.get('mode') not in ('balanced', 'mean_pca')
            or memory.get('shared_capacity') != 32 or memory.get('sketch_rank') != 32
            or any(memory.get(k) for k in ('replay', 'teacher', 'branch_B'))
            or odr.get('gpm_update_rule') != 'task_bank' or odr.get('gpm_statistics') != 'raw'
            or odr.get('gpm_max_basis') != 32 or odr.get('basis_construction') != 'projected_svd'
            or not odr.get('full_update_protection') or not odr.get('skip_orthogonal')
            or odr.get('skip_protection') != 'fused'):
        raise ValueError('Unsupported stable/task-memory protection protocol')
    bank = state.get('task_sketches')
    if not isinstance(bank, dict) or bank_hash(bank) != state.get('task_sketches_sha256'):
        raise ValueError('Missing or corrupted source task-summary SHA256')
    suffix = '.dual_gpms.global._basis'
    bases = {k[:-len(suffix)]: v for k, v in state['model'].items() if k.endswith(suffix)}
    for name, basis in bases.items():
        if (basis.ndim != 2 or not 0 < basis.shape[1] <= min(basis.shape[0], 32)
                or not torch.isfinite(basis).all()):
            raise ValueError(f'Invalid source GPM basis: {name}')
        gram = basis.float().T @ basis.float()
        if float((gram-torch.eye(gram.shape[0], device=gram.device)).abs().max()) > 1e-3:
            raise ValueError(f'Nonorthonormal source GPM basis: {name}')
    dimensions = {name: basis.shape[0] for name, basis in bases.items()}
    if not dimensions or set(bank) != set(dimensions):
        raise ValueError('Source summary layers do not match model')
    for name, entries in bank.items():
        if set(entries) != {'0', '1', '2'}:
            raise ValueError('Source must retain all three historical task summaries')
        for entry in entries.values():
            factor = entry['factor']
            if (factor.ndim != 2 or factor.shape[0] != dimensions[name]
                    or factor.shape[1] > min(dimensions[name], memory['sketch_rank'])
                    or not torch.isfinite(factor).all() or entry['columns'] < 1
                    or not math.isfinite(entry['total_energy']) or entry['total_energy'] < 0):
                raise ValueError('Invalid historical task summary')
    if set(state.get('performance_matrix', {})) != {'0', '1', '2'}:
        raise ValueError('Missing three-task evaluation history')
    for idx in range(3):
        if set(state['performance_matrix'][str(idx)]) != set(OLD_TASKS[:idx+1]):
            raise ValueError('Incomplete historical evaluation stage')


def task_memory_extension_config(source, manifest_path, output, epochs=20, batch_size=64, smoke_test=False):
    if min(epochs, batch_size) < 1:
        raise ValueError('Training budget must be positive')
    state = torch.load(source, map_location='cpu', weights_only=False)
    validate_task_memory_source(state)
    cfg = copy.deepcopy(state['run_protocol'])
    parent_path = cfg['data']['split_manifest']
    if sha256_file(parent_path) != state['split_sha256']:
        raise ValueError('Parent split SHA256 changed')
    parent = json.loads(Path(parent_path).read_text())
    cfg['data']['tasks'] += new_tasks(batch_size)
    cfg['data']['split_manifest'] = str(manifest_path)
    manifest = load_extension(cfg, parent)
    if manifest['parent_sha256'] != state['split_sha256']:
        raise ValueError('Extension parent split SHA256 mismatch')
    cfg['experiment'].update(output_dir=str(output), memory_fraction=.9)
    cfg['train'].pop('epochs_by_task', None)
    cfg['train']['epochs_per_task'] = epochs
    cfg['task_memory_extension'] = dict(
        version=1, source=str(Path(source).resolve()), source_sha256=sha256_file(source),
        source_physical_sha256=non_memory_state_fingerprint(state['model']),
        source_bank_sha256=bank_hash(state['task_sketches']), parent_split_sha256=state['split_sha256'],
        mode=cfg['task_memory']['mode'], replay=False, rebuild_old_summaries=False,
        selection='current task validation PSNR only',
        evaluation='unchanged old tasks; AWGN25 fixed noise; new tasks tile512 overlap64 / patch LPIPS',
        scope='sequence continuation from each arm own Haze4K model, not a common-weight boundary ablation')
    # Historical replication metadata remains provenance, not a new seed command.
    cfg['smoke_test'] = smoke_test
    if smoke_test:
        cfg['train'].update(epochs_per_task=1, max_steps_per_epoch=1, calibration_batches=1, gpm_batches=1)
        cfg['eval'].update(validation_interval=1, max_images=1)
        cfg['data']['num_workers'] = 0
        for task in cfg['data']['tasks'][3:]:
            task['batch_size'] = min(batch_size, 2)
    return cfg


class TaskMemoryExtensionTrainer(TaskBalancedTrainer):
    # Reuse data/evaluation only; NOT the balanced50 trainer's restore or memory rule.
    loader = FiveTaskTrainer.loader
    evaluate_task = FiveTaskTrainer.evaluate_task
    validation_for_selection = FiveTaskTrainer.validation_for_selection

    def payload(self, *args, **kwargs):
        state = super().payload(*args, **kwargs)
        state['task_memory_extension_version'] = 1
        return state

    def bootstrap_memory(self):
        raise RuntimeError('Extension must never rebuild old task summaries')

    @torch.no_grad()
    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('task_memory_extension_version') == 1:
            if state.get('task_balanced_version') != 1:
                raise ValueError('Extension checkpoint is missing its task-summary version')
            return super().restore(path)
        settings = self.cfg['task_memory_extension']
        if sha256_file(path) != settings['source_sha256']:
            raise ValueError('Must load the declared Haze4K source')
        validate_task_memory_source(state)
        if state['split_sha256'] != settings['parent_split_sha256']:
            raise ValueError('Source parent split SHA256 mismatch')
        _, _, loaded = load_checkpoint(self.model, str(path), map_location=self.device, return_state=True)
        self.model.load_state_dict(loaded['model'], strict=True)
        self.model.to(self.device)
        self.model.set_current_task(-1)
        self.task_sketches = state['task_sketches']
        self.restore_metadata(state)
        if (non_memory_fingerprint(self.model) != settings['source_physical_sha256']
                or bank_hash(self.task_sketches) != settings['source_bank_sha256']):
            raise RuntimeError('Loading extension changed weights or historical summaries')
        # Keep all original curves; only the new task gets a fresh optimizer/scheduler.
        self.history = state.get('history', [])
        self.best_psnr = self.best_epoch = self.best_path = self.frozen_sha = None
        atomic_json(Path(self.output_dir)/'haze_boundary_import.json', dict(
            source_sha256=settings['source_sha256'], physical_sha256=settings['source_physical_sha256'],
            bank_sha256=settings['source_bank_sha256'], historical_tasks=list(OLD_TASKS),
            mode=settings['mode'], no_old_train=True, gpm_reselected=False,
            note='Imported the selected arm physical weights, GPM and all frozen summaries unchanged'))
        extra = dict(in_progress=False, gpm_updated=True, selected_epoch=state.get('selected_epoch'))
        self.save(Path(self.output_dir)/'task_2_final.pth', 2, 0, **extra)
        self.save(Path(self.output_dir)/'latest.pth', 2, 0, **extra)
        return self.payload(2, 0, **extra)
