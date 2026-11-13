import torch
from torch.utils.data import BatchSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from accelerate.data_loader import BatchSamplerShard,DataLoaderShard,skip_first_batches
from p2_planning_sft.global_batch import FixedGlobalBatchShard


def rank_batches(world,rank,n=1057,skip=0):
    torch.manual_seed(20261102)
    lengths=[i*13%100+1 for i in range(n)]
    sampler=LengthGroupedSampler(batch_size=8,lengths=lengths)
    if world==32:b=BatchSamplerShard(BatchSampler(sampler,8,False),num_processes=world,process_index=rank,even_batches=True)
    else:b=FixedGlobalBatchShard(sampler,256,world,rank)
    loader=DataLoaderShard(list(range(n)),batch_sampler=b,num_workers=0,rng_types=[])
    if skip:loader=skip_first_batches(loader,skip)
    return [x.tolist() for x in loader]


def test_32_to_24_preserves_each_global_batch_and_resume_skip():
    old=[rank_batches(32,r) for r in range(32)]
    new=[rank_batches(24,r) for r in range(24)]
    for step in range(len(old[0])):
        a=[i for rank in old for i in rank[step]];b=[i for rank in new for i in rank[step]]
        assert a==b and len(a)==256
    resumed=[rank_batches(24,r,skip=2) for r in range(24)]
    for r in range(24):assert resumed[r]==new[r][2:]
    assert sorted(len(new[r][0]) for r in range(24))==[10]*8+[11]*16


def test_unequal_24_rank_token_loss_has_correct_global_gradient():
    from p2_planning_sft.model import response_loss
    from types import SimpleNamespace
    torch.manual_seed(4);logits=torch.randn(256,4,9,requires_grad=True)
    labels=torch.randint(0,9,(256,4));labels[:,0]=-100;labels[:50,1]=-100
    expected=response_loss(SimpleNamespace(logits=logits),labels)
    total=labels.ne(-100).sum();losses=[]
    for rank in range(24):
        lo=256*rank//24;hi=256*(rank+1)//24
        losses.append(24*response_loss(SimpleNamespace(logits=logits[lo:hi]),labels[lo:hi],total))
    actual=sum(losses)/24
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0])
