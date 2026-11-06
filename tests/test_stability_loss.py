import unittest

import torch

from src.trainers.stability_trainer import range_loss, replay_loss


class StabilityLossTests(unittest.TestCase):
    def test_extreme_teacher_is_bounded_and_frozen(self):
        pred = torch.tensor([-.5, .4, 1.5], requires_grad=True)
        teacher = torch.tensor([-10000., .5, 10000.], requires_grad=True)
        gt = torch.tensor([0., .5, 1.])
        settings = dict(replay_l1_weight=1., distill_weight=1., range_weight=.1)
        loss = replay_loss(pred, gt, teacher, settings)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 2.)
        self.assertIsNone(teacher.grad)
        self.assertLess(float(pred.grad[0]), 0.)
        self.assertGreater(float(pred.grad[-1]), 0.)

    def test_range_penalty_has_no_gradient_inside_unit_interval(self):
        pred = torch.tensor([.1, .9], requires_grad=True)
        range_loss(pred).backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))


if __name__ == '__main__':
    unittest.main()
