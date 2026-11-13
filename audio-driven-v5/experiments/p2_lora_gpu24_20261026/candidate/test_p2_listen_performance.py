import copy
import unittest
import torch
from torch import nn
from p2_listen.gradients import GradientAccumulator
from p2_listen.performance import BucketGradientAccumulator, configure_renderer


class PerformanceTests(unittest.TestCase):
    def test_bucket_matches_legacy_with_mixed_dtypes_and_cancellation(self):
        legacy=[nn.Parameter(torch.zeros(3,dtype=d)) for d in (torch.float32,torch.float64,torch.float32)]
        bucket=[nn.Parameter(p.detach().clone()) for p in legacy]
        a=GradientAccumulator(legacy);b=BucketGradientAccumulator(bucket,1)
        for value in (1e8,1.,-1e8):
            for parameters in (legacy,bucket):
                parameters[0].grad=torch.full_like(parameters[0],value)
                parameters[1].grad=torch.full_like(parameters[1],value)
            a.add();b.add()
        a.finish();b.finish()
        for p,q in zip(legacy,bucket):
            if p.grad is None:self.assertIsNone(q.grad)
            else:self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertEqual(bucket[0].grad[0],1.)
        self.assertIsNone(bucket[2].grad)
        prior=bucket[0].grad
        for p in bucket:p.grad=None
        bucket[1].grad=torch.zeros_like(bucket[1]);b.add();b.finish()
        self.assertTrue(torch.equal(prior,torch.ones_like(prior)))
        self.assertIsNone(bucket[0].grad)
        self.assertTrue(torch.equal(bucket[1].grad,torch.zeros_like(bucket[1])))

    def test_bucket_invalid_inputs_and_sparse_gradient(self):
        with self.assertRaises(ValueError):BucketGradientAccumulator([])
        p=nn.Parameter(torch.ones(3))
        for size in (0,-1,True,1.5):
            with self.assertRaises(ValueError):BucketGradientAccumulator([p],size)
        a=BucketGradientAccumulator([p])
        for world in (0,2):
            with self.assertRaises(ValueError):a.finish(world)
        p.grad=torch.sparse_coo_tensor(torch.tensor([[1]]),torch.tensor([1.]),(3,))
        with self.assertRaises(ValueError):a.add()

    def test_checkpoint_policies_preserve_outputs_gradients_state_and_instances(self):
        class Block(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(4,4)
            def forward(self,video,audio,perturbations):
                return (None if video is None else self.linear(video).tanh(),self.linear(audio).sin())
        class Renderer(nn.Module):
            def __init__(self):super().__init__();self.transformer_blocks=nn.ModuleList([Block() for _ in range(8)]);self.enabled=True
            def set_gradient_checkpointing(self,value):self.enabled=value
            def _process_transformer_blocks(self,video,audio,perturbations):
                for block in self.transformer_blocks:
                    if self.enabled and self.training:
                        video,audio=torch.utils.checkpoint.checkpoint(block,video,audio,perturbations,use_reentrant=False)
                    else:video,audio=block(video=video,audio=audio,perturbations=perturbations)
                return video,audio
        torch.manual_seed(9);base=Renderer();x=torch.randn(2,4)
        reference=None
        for policy in ('all','alternating','three_quarters','none'):
            model=copy.deepcopy(base);keys=set(model.state_dict());configure_renderer(model,policy)
            self.assertEqual(keys,set(model.state_dict()))
            v=x.clone().requires_grad_();a=x.clone().requires_grad_()
            out=model._process_transformer_blocks(v,a,None);sum(t.sum() for t in out).backward()
            actual=[*out,v.grad,a.grad,*[p.grad for p in model.parameters()]]
            if reference is None:reference=[t.detach().clone() for t in actual]
            for got,want in zip(actual,reference):self.assertTrue(torch.equal(got,want))
            model.eval();self.assertIsNone(model._process_transformer_blocks(None,x,None)[0])
            configure_renderer(model,'all');self.assertTrue(model.enabled)
            self.assertNotIn('_process_transformer_blocks',base.__dict__)
        with self.assertRaises(ValueError):configure_renderer(base,'invalid')
        with self.assertRaises(ValueError):configure_renderer(nn.Linear(2,2),'alternating')


if __name__=='__main__':unittest.main()
