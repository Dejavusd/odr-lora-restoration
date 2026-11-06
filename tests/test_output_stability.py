import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.scripts.audit_output_stability import coverage_hook


class StabilityCoverageTests(unittest.TestCase):
    def test_sampled_patch_response_matches_actual_convolution(self):
        generator = torch.Generator().manual_seed(19)
        x = torch.randn(1, 2, 31, 29, generator=generator)
        weight = torch.randn(3, 2, 3, 3, generator=generator)
        delta = torch.randn(3, 18, generator=generator)
        basis = torch.linalg.qr(torch.randn(18, 5, generator=generator)).Q
        layer = SimpleNamespace(padding=1, stride=2, kernel_size=3, weight=weight,
                                dual_gpms={'global': SimpleNamespace(basis=basis)})
        totals = {}
        coverage_hook('layer', delta, totals)(layer, (x,))
        patches = F.unfold(x, 3, padding=1, stride=2)[0]
        indices = torch.linspace(0, patches.shape[1]-1, 128).long()
        h = patches[:, indices]
        actual_base = F.conv2d(x, weight, padding=1, stride=2).flatten(2)[0, :, indices]
        actual_delta = F.conv2d(x, delta.reshape_as(weight), padding=1, stride=2).flatten(2)[0, :, indices]
        expected = [h.square().sum(), (basis.T@h).square().sum(),
                    actual_base.square().sum(), actual_delta.square().sum()]
        torch.testing.assert_close(torch.tensor(totals['layer']), torch.stack(expected))


if __name__ == '__main__':
    unittest.main()
