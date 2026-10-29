import copy
from pathlib import Path
import tempfile
import unittest

import torch
from src.models.odr_lora import ODRLoRAConv2d
from src.models.restoration_net import RestorationNet
from src.utils.checkpoint import save_checkpoint, load_checkpoint


class EffectiveBasisTests(unittest.TestCase):
    def test_prunes_null_shared_rows_and_normalizes_only_real_directions(self):
        layer = ODRLoRAConv2d(6, 3, 1, share_rank=2, private_rank=3,
                              rank_min=3, rank_max=3)
        layer.full_update_protection = True
        layer.basis_construction = 'projected_svd'
        layer.dual_gpms['global'].basis = torch.eye(6)[:, :2]
        layer.B_share.copy_(torch.eye(6)[:2])
        layer.share_initialized.fill_(True)
        layer.add_task()
        x = torch.randn(2, 6, 8, 8)
        x[:, 5] = 0
        layer.calibrate(x, 0)
        B = layer.task_B[0].detach()
        self.assertTrue(torch.equal(B[:2], torch.zeros_like(B[:2])))
        self.assertTrue(torch.allclose(B[2:] @ B[2:].T, torch.eye(3), atol=1e-5))
        self.assertLess(float((B @ layer.dual_gpms['global'].basis).norm()), 1e-6)
        layer.dual_gpms['global'].basis = torch.eye(6)
        layer.calibrate(x, 0)
        self.assertEqual(float(layer.task_B[0].norm()), 0)

    def test_private_rows_are_orthogonal_to_effective_shared_rows(self):
        layer = ODRLoRAConv2d(8, 3, 1, share_rank=2, private_rank=3,
                              rank_min=3, rank_max=3)
        layer.full_update_protection = True
        layer.dual_gpms['global'].basis = torch.eye(8)[:, :2]
        layer.B_share.copy_(torch.eye(8)[2:4])
        layer.add_task()
        layer._calibrate_projected(torch.randn(8, 40), 0)
        B = layer.task_B[0].detach()
        self.assertTrue(torch.allclose(B @ B.T, torch.eye(5), atol=1e-5))

    def test_variable_task_rank_checkpoint_and_optimizer_roundtrip(self):
        cfg = {
            'model': dict(base_channels=4, num_blocks=[1, 1], kernel_size=3,
                share_rank=2, private_rank_base=2, rank_gamma=1.5, rank_min=2, rank_max=8),
            'odr_lora': dict(spatial_blocks=[1, 1], spatial_sample_ratio=0.1,
                freq_level=1, freq_decomp='dwt', adaptive_lambda_min=0.5,
                adaptive_lambda_max=1., skip_orthogonal=True, share_subspace=True)}
        model = RestorationNet(cfg)
        model.add_task()
        old_shapes = {name: p.shape for name, p in model.named_parameters()
                      if '.task_A.0' in name or '.task_B.0' in name}
        model.add_task(private_rank_base=4)
        for module in model.modules():
            if isinstance(module, ODRLoRAConv2d):
                module.task_B[1].data.normal_(std=0.1)
        model.freeze_base_and_old(1)
        model.set_current_task(1)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        x = torch.randn(1, 3, 16, 16)
        model(x, 1).square().mean().backward()
        optimizer.step()
        expected = model(x, 1).detach()
        self.assertTrue(all(dict(model.named_parameters())[k].shape == shape
                            for k, shape in old_shapes.items()))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'probe.pth')
            save_checkpoint(model, optimizer, 1, 1, path)
            restored = RestorationNet(copy.deepcopy(cfg))
            restored.add_task()
            restored.add_task()  # Deliberately wrong default task-1 rank.
            restored.freeze_base_and_old(1)
            opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
            load_checkpoint(restored, path, optimizer=opt)
        self.assertTrue(torch.allclose(restored(x, 1), expected, atol=1e-5))
        trainable = {p for p in restored.parameters() if p.requires_grad}
        self.assertEqual(set(opt.param_groups[0]['params']), trainable)
        opt.zero_grad()
        restored(x, 1).square().mean().backward()
        opt.step()


if __name__ == '__main__':
    unittest.main()
