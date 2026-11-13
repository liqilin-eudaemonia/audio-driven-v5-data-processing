import copy
import pytest
import torch
from torch import nn
from p2_legacy_execution import configure,select_policy

class Block(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.randn(3,3))
    def forward(self,video,audio,perturbations=None):return torch.sin(video@self.weight),torch.cos(audio@self.weight)
class Renderer(nn.Module):
    def __init__(self):super().__init__();self.transformer_blocks=nn.ModuleList([Block() for _ in range(6)])
    def set_gradient_checkpointing(self,value):self.enabled=value
    def _process_transformer_blocks(self,video,audio,perturbations):
        from torch.utils.checkpoint import checkpoint
        for b in self.transformer_blocks:
            if self.enabled and self.training:video,audio=checkpoint(b,video,audio,perturbations,use_reentrant=False)
            else:video,audio=b(video=video,audio=audio,perturbations=perturbations)
        return video,audio
    def forward(self,x):return self._process_transformer_blocks(x,x,None)

def test_adaptive_switch_preserves_gradients_and_parameters():
    torch.manual_seed(7);base=Renderer();a=copy.deepcopy(base);b=copy.deepcopy(base)
    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    for policy in ['none','alternating','two_thirds','three_quarters','none','two_thirds']:
        x=torch.randn(1,3)
        for m,o,p in [(a,oa,'all'),(b,ob,policy)]:
            configure(m,p);o.zero_grad();values=m(x);sum(v.square().mean() for v in values).backward();o.step()
        for u,v in zip(a.parameters(),b.parameters()):assert torch.equal(u,v)

@pytest.mark.parametrize('seconds,expected',[(2.,'none'),(5.5,'none'),(5.501,'alternating'),(10.,'alternating'),(10.001,'two_thirds'),(12.,'two_thirds')])
def test_duration_boundaries(seconds,expected):
    assert select_policy(seconds,[dict(max_duration_s=5.5,policy='none'),dict(max_duration_s=10.,policy='alternating'),dict(max_duration_s=12.,policy='two_thirds')])==expected

@pytest.mark.parametrize('seconds',[0,-1,float('nan'),float('inf'),12.001])
def test_reject_unbenchmarked_duration(seconds):
    with pytest.raises(ValueError):select_policy(seconds,[dict(max_duration_s=12,policy='two_thirds')])
