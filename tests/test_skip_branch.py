import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.models.backbone import PixelChannelNorm, normalized_skip_components
from src.models.odr_lora import ODRLoRAConv2d
from src.models.restoration_net import RestorationNet
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume
from src.trainers.memory_source_trainer import non_memory_fingerprint
from src.trainers.skip_branch_trainer import skip_branch_config, SkipBranchTrainer
from tests import test_clean_sequence as fixtures


class SkipBranchTests(unittest.TestCase):
    def test_normalized_components_use_fused_statistics(self):
        for norm in (PixelChannelNorm(4), torch.nn.GroupNorm(1, 4),
                     torch.nn.GroupNorm(1, 4, affine=False)):
            d, s = torch.randn(2, 4, 9, 11), torch.randn(2, 4, 9, 11)
            if getattr(norm, 'weight', None) is not None:
                with torch.no_grad():
                    norm.weight.uniform_(.5, 1.5)
                    norm.bias.uniform_(-.3, .3)
            a, b, actual = normalized_skip_components(norm, d, s)
            torch.testing.assert_close(a+b, norm(d+s), atol=1e-6, rtol=1e-5)
            self.assertFalse(a.requires_grad)
            self.assertFalse(torch.allclose(norm(d)+norm(s), actual))
        # A cancelled fused feature can hide significant opposing branch components.
        d = torch.randn(1, 4, 7, 9)
        a, b, actual = normalized_skip_components(PixelChannelNorm(4), d, -d)
        self.assertGreater(float(a.norm()), 1.)
        torch.testing.assert_close(a+b, actual)

    def test_matched_columns_and_cancellation_signal(self):
        layer = ODRLoRAConv2d(1, 1, 1, freq_decomp='none', spatial_blocks=(1, 1),
                              spatial_sample_ratio=1.)
        layer.gpm_statistics = 'raw'
        layer.gpm_update_rule = 'bounded_svd'
        layer.gpm_sample_columns = 16
        x = torch.zeros(1, 1, 8, 8)
        calls = []
        for branches in (None, (torch.ones_like(x), -torch.ones_like(x))):
            with patch.object(layer.dual_gpms['global'], 'update') as update:
                layer.update_gpm(x, 0, branches=branches)
                calls.append(update.call_args.args[0])
        self.assertEqual(calls[0].shape, calls[1].shape)
        self.assertEqual(calls[1].shape[1], 16)
        self.assertEqual(float(calls[0].square().sum()), 0.)
        self.assertGreater(float(calls[1].square().sum()), 0.)
        self.assertEqual(layer.skip_column_history[-1]['fused'], 8)
        self.assertEqual(layer.skip_column_history[-1]['decoder'], 4)
        self.assertEqual(layer.skip_column_history[-1]['skip'], 4)

    def test_actual_targets_forward_rank_and_non_targets_unchanged(self):
        fixture = fixtures.CleanSequenceTests()
        fixture.setUp()
        try:
            cfg = fixture.config_at('targets')
            cfg['model']['num_blocks'] = [2, 2, 2]
            cfg['odr_lora'].update(gpm_statistics='raw', gpm_update_rule='bounded_svd')
            for architecture in ('legacy', 'stable_v1'):
                cfg['model']['architecture'] = architecture
                torch.manual_seed(42)
                fused = RestorationNet(cfg)
                other = copy.deepcopy(cfg)
                other['odr_lora']['skip_protection'] = 'branch_v1'
                torch.manual_seed(42)
                branch = RestorationNet(other)
                self.assertEqual(non_memory_fingerprint(fused), non_memory_fingerprint(branch))
                for m in (fused, branch):
                    m.add_task()
                x = torch.rand(2, 3, 17, 19)
                torch.testing.assert_close(fused(x), branch(x), rtol=0, atol=0)
                torch.manual_seed(17); fused.update_gpm(x, 0)
                torch.manual_seed(17); branch.update_gpm(x, 0)
                targets = []
                fused_modules = dict(fused.named_modules())
                for name, layer in branch.named_modules():
                    if not isinstance(layer, ODRLoRAConv2d):
                        continue
                    before = fused_modules[name]
                    self.assertEqual(layer.gpm_column_history, before.gpm_column_history)
                    self.assertEqual(layer.task_A[0].shape, before.task_A[0].shape)
                    if layer.skip_column_history:
                        targets.append(name)
                    else:
                        torch.testing.assert_close(layer.dual_gpms['global'].basis,
                                                   before.dual_gpms['global'].basis, rtol=0, atol=0)
                    self.assertLessEqual(layer.dual_gpms['global'].basis.shape[1], 2)
                self.assertEqual(targets, ['backbone.dec_blocks.1.0.conv1',
                                           'backbone.dec_blocks.2.0.conv1'])
                self.assertIsNone(branch.backbone._skip_branch_capture)
                torch.testing.assert_close(fused(x), branch(x), rtol=0, atol=0)
                branch.add_task(); branch.set_current_task(1); branch.calibrate(x, 1)
                for layer in branch.modules():
                    if isinstance(layer, ODRLoRAConv2d):
                        basis = layer.dual_gpms['global'].basis
                        self.assertLess(float((layer.task_B[1] @ basis).norm()), 1e-4)
        finally:
            fixture.tearDown()

    def test_invalid_statistics_and_branch_inputs_rejected(self):
        layer = ODRLoRAConv2d(1, 1, 1)
        x = torch.zeros(1, 1, 8, 8)
        with self.assertRaisesRegex(ValueError, 'requires raw'):
            layer.update_gpm(x, 0, (x, x))
        layer.gpm_statistics = 'raw'; layer.gpm_update_rule = 'bounded_svd'
        with self.assertRaisesRegex(RuntimeError, 'reconstruct'):
            layer.update_gpm(x, 0, (x+1, x))

    def test_common_source_no_replay_exact_resume_and_protocol(self):
        fixture = fixtures.CleanSequenceTests()
        fixture.setUp()
        try:
            cfg = fixture.config_at('origin')
            cfg['model']['architecture'] = 'stable_v1'
            cfg['train']['epochs_per_task'] = 1
            cfg['train']['epochs_by_task'] = dict(lowlight=1, rainh=99)
            original = CleanSequenceTrainer(cfg, fixture.manifest)
            original.run(stop_after='lowlight')
            source = Path(original.output_dir)/'task_0_final.pth'

            class Guarded(SkipBranchTrainer):
                bootstrap = False
                active_task = None
                def bootstrap_memory(self):
                    self.bootstrap = True
                    try:
                        return super().bootstrap_memory()
                    finally:
                        self.bootstrap = False
                def train_task(self, task, *args, **kwargs):
                    self.active_task = task
                    try:
                        return super().train_task(task, *args, **kwargs)
                    finally:
                        self.active_task = None
                def loader(self, task, role):
                    if role == 'train' and not (self.bootstrap and task == 'lowlight'):
                        if task != self.active_task:
                            raise AssertionError('Old TRAIN loader accessed')
                    return super().loader(task, role)

            fingerprints, budgets = [], []
            for mode in ('fused', 'branch_v1'):
                config = skip_branch_config(source, fixture.root/mode, mode, epochs=2, batch_size=2)
                config['eval']['validation_interval'] = 1
                self.assertNotIn('epochs_by_task', config['train'])
                trainer = Guarded(config, fixture.manifest)
                trainer.restore(source)
                fingerprints.append(non_memory_fingerprint(trainer.model))
                budgets.append(json.loads((Path(trainer.output_dir)/'task_0_memory_bootstrap.json').read_text())['columns'])
            self.assertEqual(fingerprints[0], fingerprints[1])
            self.assertEqual(fingerprints[0], non_memory_fingerprint(original.model))
            self.assertEqual(budgets[0], budgets[1])
            config = skip_branch_config(source, fixture.root/'reference', epochs=2, batch_size=2)
            config['eval']['validation_interval'] = 1
            reference = Guarded(config, fixture.manifest)
            initial = fixture.root/'branch_v1/task_0_final.pth'
            reference.run(resume_path=initial)
            interrupted_cfg = copy.deepcopy(config)
            interrupted_cfg['experiment']['output_dir'] = str(fixture.root/'interrupted')
            interrupted = Guarded(interrupted_cfg, fixture.manifest)
            save = interrupted.save
            def disconnect(path, *args, **kwargs):
                save(path, *args, **kwargs)
                if Path(path).name == 'latest.pth':
                    raise RuntimeError('disconnect')
            interrupted.save = disconnect
            with self.assertRaisesRegex(RuntimeError, 'disconnect'):
                interrupted.run(resume_path=initial)
            latest = fixture.root/'interrupted/latest.pth'
            state = torch.load(latest, weights_only=False)
            changed = copy.deepcopy(interrupted_cfg)
            changed['odr_lora']['skip_protection'] = 'fused'
            with self.assertRaisesRegex(ValueError, '训练协议'):
                validate_resume(changed, state, state['split_sha256'])
            resumed = Guarded(interrupted_cfg, fixture.manifest)
            resumed.run(resume_path=latest)
            self.assertEqual(reference.history, resumed.history)
            for key, value in reference.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
