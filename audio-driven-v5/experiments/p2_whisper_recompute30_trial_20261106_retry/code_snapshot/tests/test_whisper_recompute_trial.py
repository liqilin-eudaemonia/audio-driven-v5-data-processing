import copy
from unittest.mock import patch
import torch
import pytest
from test_p2_whisper_ablation import build
from p2_listen.performance import configure_renderer
from p2_matrix16.training import objective


def test_three_quarters_skips_a_real_block_and_matches_two_updates():
    from ltx_core.model.transformer.model import LTXModel
    original=LTXModel.__init__
    def four_blocks(self,*a,**kw):
        kw['num_layers']=4
        original(self,*a,**kw)
    with patch.object(LTXModel,'__init__',four_blocks):c,m,_,bundle,strategy,sampler=build('event')
    assert len(m.renderer.transformer_blocks)==4
    from test_p2_whisper_av import opened
    opened(m);initial=copy.deepcopy(m.state_dict());results=[]
    for policy in ['all','three_quarters','all']:
        m.load_state_dict(initial);configure_renderer(m.renderer,policy);m.train();opt=torch.optim.AdamW(m.groups())
        losses=[];calls=[]
        hook=m.renderer.transformer_blocks[3].register_forward_pre_hook(lambda *args:calls.append(1))
        for i in range(2):
            opt.zero_grad(set_to_none=True);torch.manual_seed(83+i)
            v=m(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(93+i),strategy,sampler)
            loss=objective(v,c,{k:1. for k in bundle['events']},1,1)[0];loss.backward();losses.append(float(loss.detach()));opt.step()
        hook.remove();assert len(calls)==(2 if policy=='three_quarters' else 4)
        results.append((copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),losses))
    from p2_j_event_feature_continue24 import equal
    assert equal(results[0],results[1]) and equal(results[0],results[2])
