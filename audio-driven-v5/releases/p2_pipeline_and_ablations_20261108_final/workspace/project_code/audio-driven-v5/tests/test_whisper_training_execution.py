import torch
import pytest

from p2_whisper_av.execution import StreamingFP64Reduction,move_adam_moments,settings
from p2_listen.performance import BucketGradientAccumulator


def test_explicit_execution_settings_override_defaults():
    explicit=dict(gradient_reduction='full_fp64',gradient_bucket_mb=16,
                  optimizer_state_cpu_offload=False,longest_sample_check=False)
    assert settings({'whisper_training_execution':explicit})==explicit
    assert settings({'whisper_training_execution':{'gradient_bucket_mb':64}})['gradient_bucket_mb']==64
    with pytest.raises(ValueError,match='Invalid Whisper'):
        settings({'whisper_training_execution':{'gradient_bucket_mb':0}})


def test_streaming_matches_full_fp64_and_keeps_unused_none_across_buckets():
    torch.manual_seed(19)
    left=[torch.nn.Parameter(torch.randn(150000)),torch.nn.Parameter(torch.randn(3)),torch.nn.Parameter(torch.randn(140000))]
    right=[torch.nn.Parameter(p.detach().clone()) for p in left]
    for a,b in zip(left,right):
        a.grad=torch.randn_like(a);b.grad=a.grad.clone()
    left[1].grad=right[1].grad=None
    full=BucketGradientAccumulator(left,1);small=StreamingFP64Reduction(right,1)
    full.add();small.add();full.finish();small.finish()
    for a,b in zip(left,right):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:assert torch.equal(a.grad,b.grad)
    assert not hasattr(small,'buffer')


def test_streaming_rejects_multiple_local_microbatches():
    p=torch.nn.Parameter(torch.ones(2));p.grad=torch.ones(2)
    x=StreamingFP64Reduction([p]);x.add()
    with pytest.raises(ValueError,match='one microbatch'):x.add()


def test_adam_moment_transfer_preserves_updates_and_checkpoint(tmp_path):
    a=torch.nn.Parameter(torch.randn(8));b=torch.nn.Parameter(a.detach().clone())
    oa=torch.optim.AdamW([a],lr=.001);ob=torch.optim.AdamW([b],lr=.001)
    for _ in range(3):
        for p,o in [(a,oa),(b,ob)]:
            o.zero_grad();p.square().sum().backward()
            if o is ob:move_adam_moments(o,to_parameters=True)
            o.step()
            if o is ob:move_adam_moments(o)
        assert torch.equal(a,b)
        for key in oa.state[a]:assert torch.equal(oa.state[a][key],ob.state[b][key])
    assert settings({})['optimizer_state_cpu_offload']
