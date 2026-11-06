import copy
import unittest

import torch
import yaml
from pathlib import Path

from src.models.backbone import PixelChannelNorm, BoundedGate, NAFBlock
from src.models.restoration_net import RestorationNet
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume
from tests import test_clean_sequence as fixtures


class StableBackboneTests(unittest.TestCase):
    def test_pixel_norm_is_local(self):
        norm = PixelChannelNorm(4)
        x = torch.randn(2, 4, 9, 11, requires_grad=True)
        torch.testing.assert_close(norm(x)[..., :4, :5], norm(x[..., :4, :5]))
        norm(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_gate_magnitude(self):
        x = torch.randn(2, 8, 9, 11) * 100
        self.assertTrue((BoundedGate()(x).abs() <= x[:, :4].abs()).all())

    def test_default_legacy_unchanged(self):
        cfg = yaml.safe_load(Path('configs/clean_lsrw_rainh_haze4k.yaml').read_text())
        cfg['model'].update(base_channels=4, num_blocks=[1, 1])
        torch.manual_seed(1)
        a = RestorationNet(cfg)
        cfg['model']['architecture'] = 'legacy'
        torch.manual_seed(1)
        b = RestorationNet(cfg)
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[name]), name)
        x = torch.rand(1, 3, 17, 19)
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    def test_stable_epoch_resume(self):
        fixture = fixtures.CleanSequenceTests()
        fixture.setUp()
        try:
            fixture.cfg['model']['architecture'] = 'stable_v1'
            fixture.test_epoch_resume_matches_uninterrupted_training()
        finally:
            fixture.tearDown()

    def test_stable_incremental_merge_and_protocol(self):
        fixture = fixtures.CleanSequenceTests()
        fixture.setUp()
        try:
            cfg = fixture.config_at('stable')
            cfg['model']['architecture'] = 'stable_v1'
            cfg['odr_lora'].update(gpm_statistics='raw', gpm_update_rule='bounded_svd')
            cfg['train']['epochs_per_task'] = 2
            cfg['train']['epochs_by_task'] = dict(lowlight=1, rainh=1, dehaze=1)
            trainer = CleanSequenceTrainer(cfg, fixture.manifest)
            trainer.run()
            blocks = [b for b in trainer.model.modules() if isinstance(b, NAFBlock)]
            self.assertTrue(all(isinstance(b.norm1, PixelChannelNorm) and b.bounded_residual
                                and b.sca.bounded for b in blocks))
            state = torch.load(fixture.root/'stable'/'task_2_final.pth', weights_only=False)
            self.assertEqual(state['selected_epoch'], 1)
            changed = copy.deepcopy(cfg)
            changed['model']['architecture'] = 'legacy'
            with self.assertRaises(ValueError):
                validate_resume(changed, state, state['split_sha256'])
            for p in trainer.model.parameters():
                self.assertTrue(torch.isfinite(p).all())
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
