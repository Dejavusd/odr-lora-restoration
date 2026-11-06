import unittest

import torch

from src.models.odr_lora import ODRLoRAConv2d
from src.trainers.bounded_lora_trainer import bound_updates


class BoundProjectionTests(unittest.TestCase):
    def model(self):
        layer = ODRLoRAConv2d(2,2,1,share_rank=0,private_rank=1,rank_min=1,rank_max=1)
        layer.add_task()
        with torch.no_grad():
            layer.weight.fill_(1.)
            layer.task_A[0].fill_(10.)
            layer.task_B[0].zero_()
            layer.task_B[0][:,1] = 1.
        layer.dual_gpms['global'].basis = torch.tensor([[1.],[0.]])
        return torch.nn.Sequential(layer), layer

    def test_cap_preserves_orthogonality_and_frozen_tensors(self):
        model,layer = self.model()
        weight,basis,b = layer.weight.clone(),layer.dual_gpms['global'].basis.clone(),layer.task_B[0].clone()
        stats = bound_updates(model,0,.05)
        self.assertEqual(stats['clipped_layers'],1)
        self.assertLessEqual(stats['max_after'],.05)
        torch.testing.assert_close(layer.weight,weight,rtol=0,atol=0)
        torch.testing.assert_close(layer.task_B[0],b,rtol=0,atol=0)
        torch.testing.assert_close(layer.dual_gpms['global'].basis,basis,rtol=0,atol=0)
        self.assertEqual(float(((layer.task_A[0]@layer.task_B[0])@basis).norm()),0.)

    def test_disabled_bound_and_idempotence(self):
        model,layer = self.model()
        original = layer.task_A[0].clone()
        bound_updates(model,0,0.)
        torch.testing.assert_close(layer.task_A[0],original,rtol=0,atol=0)
        bound_updates(model,0,.05)
        bounded = layer.task_A[0].clone()
        stats = bound_updates(model,0,.05)
        self.assertEqual(stats['clipped_layers'],0)
        torch.testing.assert_close(layer.task_A[0],bounded,rtol=0,atol=0)

    def test_zero_weight_and_nonfinite_update(self):
        model,layer = self.model()
        with torch.no_grad():
            layer.weight.zero_()
        stats = bound_updates(model,0,.05)
        self.assertLessEqual(stats['max_after'],.05)
        with torch.no_grad():
            layer.task_A[0].fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            bound_updates(model,0,.05)


if __name__ == '__main__':
    unittest.main()
