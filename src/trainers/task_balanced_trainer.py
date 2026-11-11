"""Per-task frozen summaries, fixed shared GPM rank, no replay or branch B."""
from pathlib import Path

import torch

from src.models.odr_lora import ODRLoRAConv2d
from src.models.task_memory import TaskSketchAccumulator, select_task_memory, bank_hash
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume
from src.trainers.memory_source_trainer import MemorySourceTrainer, memory_source_config
from src.utils.training_state import atomic_json


def task_balanced_config(source, output, mode='balanced', epochs=20, batch_size=64, smoke_test=False):
    if mode not in ('balanced', 'mean_pca'):
        raise ValueError('Unknown task-memory mode')
    cfg = memory_source_config(source, output, 'raw', epochs, batch_size, smoke_test)
    if cfg['odr_lora'].get('basis_construction') != 'projected_svd':
        raise ValueError('Task memory requires projected_svd')
    cfg['train'].pop('epochs_by_task', None)
    cfg['odr_lora'].update(skip_protection='fused', skip_orthogonal=True, gpm_update_rule='task_bank')
    cfg['task_memory'] = dict(version=1, mode=mode, sketch_rank=32, replay=False, teacher=False,
        branch_B=False, shared_capacity=32, per_task_normalization='retained sketch trace = 1',
        selector='worst normalized sketch coverage first' if mode == 'balanced' else 'equal-task mean covariance PCA',
        sketch='streaming rank32 truncated SVD of current-task TRAIN only',
        budget='shared protected rank fixed; extra frozen summaries grow linearly with task count',
        selection='new task validation PSNR only',
        limitation='historical snapshots are stale; sketch coverage is not actual old-task output preservation')
    return cfg


class TaskBalancedTrainer(MemorySourceTrainer):
    def __init__(self, *args, **kwargs):
        self.task_sketches = {}
        super().__init__(*args, **kwargs)

    def payload(self, *args, **kwargs):
        result = super().payload(*args, **kwargs)
        result.update(task_balanced_version=1, task_sketches=self.task_sketches,
                      task_sketches_sha256=bank_hash(self.task_sketches))
        return result

    def _begin_collection(self):
        for m in self.model.modules():
            if isinstance(m, ODRLoRAConv2d):
                m._task_sketch_accumulator = TaskSketchAccumulator(m.d, self.cfg['task_memory']['sketch_rank'])

    def _clear_collection(self):
        for m in self.model.modules():
            if isinstance(m, ODRLoRAConv2d):
                m._task_sketch_accumulator = None

    def bootstrap_memory(self):
        self.task_sketches = {}
        self._begin_collection()
        try:
            return super().bootstrap_memory()
        finally:
            self._clear_collection()

    def _finish_task(self, loader, idx):
        self._begin_collection()
        try:
            return super()._finish_task(loader, idx)
        finally:
            self._clear_collection()

    @torch.no_grad()
    def finalize_memory(self, idx):
        layers = {}
        key = str(idx)
        for name, m in self.model.named_modules():
            if not isinstance(m, ODRLoRAConv2d):
                continue
            bank = self.task_sketches.setdefault(name, {})
            if key in bank:
                raise ValueError('Task summary already finalized; do not overwrite historical snapshots')
            if set(bank) != {str(i) for i in range(idx)}:
                raise ValueError('Missing historical task summaries')
            bank[key] = m._task_sketch_accumulator.finish()
            memory = m.dual_gpms['global']
            basis, info = select_task_memory(bank, memory.max_basis, self.cfg['task_memory']['mode'])
            memory.basis = basis.to(device=m.weight.device, dtype=m.weight.dtype)
            memory.singular_values = None
            m._task_sketch_accumulator = None
            layers[name] = info
            if len(layers) % 30 == 0:
                self.logger.info(f'TASK memory {len(layers)} layers, {idx+1} task summaries/layer')
        sketch_bytes = sum(sum(v['sketch_bytes'] for v in row['tasks'].values()) for row in layers.values())
        protected_bytes = sum(m.dual_gpms['global'].basis.numel()*m.dual_gpms['global'].basis.element_size()
                              for m in self.model.modules() if isinstance(m, ODRLoRAConv2d))
        atomic_json(Path(self.output_dir)/f'task_{idx}_task_memory.json', dict(
            mode=self.cfg['task_memory']['mode'], task_count=idx+1, shared_capacity=32,
            sketch_bytes=sketch_bytes, protected_basis_bytes=protected_bytes,
            total_tensor_bytes=sketch_bytes+protected_bytes,
            bank_sha256=bank_hash(self.task_sketches), layers=layers,
            note='Frozen per-task summaries; no raw images, gradients or old-task TRAIN reads. '
                 'Coverage is on retained snapshots, not current old features; extra storage grows with tasks.'))

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('task_balanced_version') != 1:
            return super().restore(path)
        validate_resume(self.cfg, state, self.split_sha)
        bank = state.get('task_sketches')
        if not isinstance(bank, dict) or bank_hash(bank) != state.get('task_sketches_sha256'):
            raise ValueError('Task-summary bank is missing or its SHA256 is invalid')
        count = state['task_idx'] + (0 if state.get('in_progress') else 1)
        layers = {n: m for n, m in self.model.named_modules() if isinstance(m, ODRLoRAConv2d)}
        if set(bank) != set(layers):
            raise ValueError('Task-summary layer names do not match model')
        for name, entries in bank.items():
            if set(entries) != {str(i) for i in range(count)}:
                raise ValueError('Checkpoint has incomplete per-task summaries')
            for entry in entries.values():
                factor = entry['factor']
                if (factor.ndim != 2 or factor.shape[0] != layers[name].d
                        or factor.shape[1] > min(layers[name].d, self.cfg['task_memory']['sketch_rank'])
                        or not torch.isfinite(factor).all()):
                    raise ValueError('Invalid task-summary shape or values')
        self.task_sketches = bank
        return CleanSequenceTrainer.restore(self, path)
