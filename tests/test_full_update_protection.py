import tempfile
import unittest
from pathlib import Path

import torch
from src.models.odr_lora import ODRLoRAConv2d
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class ProtectionTests(unittest.TestCase):
    def layer(self):
        layer = ODRLoRAConv2d(2, 3, 3, share_rank=2, private_rank=2,
                              rank_min=2, rank_max=4)
        layer.add_task()
        layer.dual_gpms['global'].basis = torch.linalg.qr(torch.randn(18, 4)).Q
        return layer

    def test_shared_and_private_updates_stay_protected_after_optimizer_steps(self):
        torch.manual_seed(4)
        layer = self.layer()
        layer.task_B[0].data.normal_()
        layer.protect_task_basis(0)
        layer.freeze_base_and_old(0)
        optimizer = torch.optim.AdamW([layer.task_A[0]], lr=0.01)
        x = torch.randn(2, 2, 12, 12)
        for _ in range(4):
            optimizer.zero_grad()
            layer(x, 0).square().mean().backward()
            optimizer.step()
        delta = layer.task_A[0] @ layer.task_B[0]
        self.assertGreater(float(delta.norm()), 0)
        self.assertLess(float((delta @ layer.dual_gpms['global'].basis).norm()), 1e-5)
        before = layer(x, 0).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'probe.pth')
            save_checkpoint(layer, optimizer, 4, 0, path)
            restored = self.layer()
            load_checkpoint(restored, path)
            self.assertTrue(torch.allclose(before, restored(x, 0), atol=1e-6))
        layer.merge_task(0)
        self.assertTrue(torch.allclose(before, layer(x, -1), atol=2e-6))

    def test_full_rank_memory_leaves_no_update_capacity(self):
        layer = self.layer()
        layer.dual_gpms['global'].basis = torch.eye(18)
        layer.task_B[0].data.normal_()
        layer.protect_task_basis(0)
        self.assertEqual(float(layer.task_B[0].norm()), 0)

    def test_calibration_projects_shared_and_private_rows_when_enabled(self):
        layer = self.layer()
        layer.full_update_protection = True
        layer.calibrate(torch.randn(2, 2, 16, 16), 0, task_similarity=1.0)
        self.assertLess(float((layer.task_B[0] @ layer.dual_gpms['global'].basis).norm()), 1e-5)


if __name__ == '__main__':
    unittest.main()
