import unittest
import torch
import torch.nn.functional as F

from src.models.dual_gpm import DualGPM
from src.models.odr_lora import ODRLoRAConv2d


class RawGPMTests(unittest.TestCase):
    def test_capacity_extension_preserves_old_span_and_is_nested(self):
        memory = DualGPM(5, eps=0.99, max_basis=2)
        old = torch.eye(5)[:, :1]
        memory.basis = old.clone()
        C = torch.diag(torch.tensor([1., 10., 5., .1, .01]))
        small, a = memory.extend_from_covariance(C, max_basis=2)
        large, b = memory.extend_from_covariance(C, max_basis=3)
        self.assertTrue(torch.equal(memory.basis, old))
        self.assertLess(float((old-small@(small.T@old)).norm()), 1e-6)
        self.assertLess(float((small-large@(large.T@small)).norm()), 1e-6)
        self.assertTrue(torch.allclose(large.T@large, torch.eye(3), atol=1e-6))
        self.assertGreater(b['sampled_coverage_after'], a['sampled_coverage_after'])
        with self.assertRaises(ValueError):
            memory.basis = large
            memory.extend_from_covariance(C, max_basis=2)

    def test_raw_coordinates_match_convolution(self):
        layer = ODRLoRAConv2d(2, 3, 3, stride=1, padding=1)
        x = torch.randn(2, 2, 8, 8)
        H = layer.raw_input_matrix(x)
        unfolded = (layer.weight.flatten(1) @ H).reshape(3, 2, 8, 8).permute(1, 0, 2, 3)
        expected = F.conv2d(x, layer.weight, padding=1)
        self.assertTrue(torch.allclose(unfolded, expected, atol=1e-5))

    def test_raw_update_does_not_drop_previous_directions(self):
        memory = DualGPM(6, max_basis=4)
        memory.basis = torch.eye(6)[:, :2]
        old = memory.basis.clone()
        memory.update_preserving(torch.randn(6, 100))
        self.assertLess(float((old-memory.project_in(old)).norm()), 1e-6)
        self.assertLessEqual(memory.basis.shape[1], 4)

    def test_convolution_raw_policy_preserves_existing_memory(self):
        layer = ODRLoRAConv2d(2, 3, 3, share_rank=2, private_rank=2,
                              rank_min=2, rank_max=4)
        layer.gpm_statistics = 'raw'
        layer.gpm_sample_columns = 32
        memory = layer.dual_gpms['global']
        memory.basis = torch.eye(18)[:, :1]
        old = memory.basis.clone()
        layer.update_gpm(torch.randn(2, 2, 8, 8), 0)
        self.assertLess(float((old-memory.project_in(old)).norm()), 1e-6)
        self.assertLessEqual(memory.basis.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
