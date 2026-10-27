import unittest
import torch
from p2_listen.gradients import GradientAccumulator


class StableGradients(unittest.TestCase):
    def test_small_signal_survives_cancellation_before_rounding_to_parameter_dtype(self):
        p=torch.nn.Parameter(torch.zeros(1));acc=GradientAccumulator([p])
        for v in (1e8,1.,-1e8):
            p.grad=torch.tensor([v]);acc.add();self.assertIsNone(p.grad)
        acc.finish()
        self.assertEqual(p.grad.dtype,torch.float32)
        self.assertEqual(p.grad.item(),1.)
        naive=torch.tensor(1e8)+torch.tensor(1.)+torch.tensor(-1e8)
        self.assertEqual(naive.item(),0.)

    def test_unused_and_zero_gradient_are_distinct(self):
        used=torch.nn.Parameter(torch.ones(2));unused=torch.nn.Parameter(torch.ones(2))
        acc=GradientAccumulator([used,unused]);used.grad=torch.zeros_like(used);acc.add();acc.finish()
        self.assertIsNone(unused.grad)
        self.assertTrue(torch.equal(used.grad,torch.zeros_like(used)))

    def test_does_not_mutate_parameters_and_resets_for_next_step(self):
        p=torch.nn.Parameter(torch.tensor([3.]));acc=GradientAccumulator([p])
        for v in (1.,2.):p.grad=torch.tensor([v]);acc.add()
        acc.finish();self.assertEqual(p.item(),3.);self.assertEqual(p.grad.item(),3.)
        p.grad=None;acc.add();acc.finish();self.assertIsNone(p.grad)

    def test_world_mismatch_fails(self):
        acc=GradientAccumulator([torch.nn.Parameter(torch.ones(1))])
        with self.assertRaises(ValueError):acc.finish(2)
        with self.assertRaises(ValueError):acc.finish(0)


if __name__=='__main__':unittest.main()
