"""Matched skip-branch GPM experiment; no extra parameters, replay or rank change."""
from pathlib import Path

from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.memory_source_trainer import memory_source_config, MemorySourceTrainer
from src.utils.training_state import atomic_json


def skip_branch_config(source, output, mode='branch_v1', epochs=20, batch_size=64, smoke_test=False):
    if mode not in ('fused', 'branch_v1'):
        raise ValueError('Unknown skip memory mode')
    cfg = memory_source_config(source, output, 'raw', epochs, batch_size, smoke_test)
    if cfg['odr_lora'].get('basis_construction') != 'projected_svd':
        raise ValueError('Source must use projected_svd effective bases')
    # Task-specific 400/20/20 source budgets must not override this short experiment.
    cfg['train'].pop('epochs_by_task', None)
    cfg['odr_lora'].update(skip_protection=mode, skip_orthogonal=True)
    cfg['skip_branch'] = dict(version=1, mode=mode, replay=False, teacher=False,
        target='first conv1 of first decoder block immediately after each encoder skip addition',
        coordinates='additive contributions under fused normalization statistics; not norm(d)+norm(s)',
        sampling='same total columns; fused half, decoder quarter, skip quarter; remainder to fused',
        capacity='same min(32,d) shared GPM; no separate branch memories',
        rank='unchanged source scale-dependent LoRA ranks',
        selection='current task validation PSNR only')
    return cfg


class SkipBranchTrainer(MemorySourceTrainer):
    def payload(self, *args, **kwargs):
        result = super().payload(*args, **kwargs)
        result['skip_branch_version'] = 1
        return result

    def save_branch_diagnostics(self, idx):
        targets = {}
        for level in range(1, self.model.backbone.num_levels):
            blocks = self.model.backbone.dec_blocks[level]
            if len(blocks):
                name = f'backbone.dec_blocks.{level}.0.conv1'
                layer = blocks[0].conv1
                memory = layer.dual_gpms['global']
                targets[name] = dict(columns=list(layer.gpm_column_history),
                    branch_samples=list(layer.skip_column_history), cap=memory.max_basis,
                    rank=0 if memory.basis is None else memory.basis.shape[1])
        atomic_json(Path(self.output_dir)/f'task_{idx}_skip_memory.json', dict(
            mode=self.cfg['skip_branch']['mode'], targets=targets,
            note='GPM only; actual network forward, adapter placement and ranks unchanged'))

    def clear_memory_counts(self):
        super().clear_memory_counts()
        for m in self.model.modules():
            if isinstance(m, ODRLoRAConv2d):
                m.skip_column_history = []

    def bootstrap_memory(self):
        self.clear_memory_counts()
        super().bootstrap_memory()
        self.save_branch_diagnostics(0)

    def _finish_task(self, loader, idx):
        super()._finish_task(loader, idx)
        self.save_branch_diagnostics(idx)
