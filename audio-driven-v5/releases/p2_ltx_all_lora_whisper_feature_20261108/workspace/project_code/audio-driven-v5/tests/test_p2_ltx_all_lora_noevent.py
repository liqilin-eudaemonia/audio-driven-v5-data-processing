"""No-Event expanded LoRA: real native forward, gradients and exact Adam resume."""
import copy
from unittest.mock import patch
import pytest,torch
from p2_listen.contracts import read
from p2_ltx_all_lora_noevent.config import expected,validate
from p2_ltx_all_lora_noevent import checkpoint
from p2_ltx_all_lora_noevent.model import Model
from p2_ltx_all_lora.lora import ATTENTION,FFN,WHISPER
from p2_ltx_events.heads import objective,names
from p2_listen.training import rng_state
from test_p2_current_component_ablation import build as baseline_build,forward
from test_p2_whisper_av import tiny,opened

torch.set_num_threads(2)

def fixture():
 c,cfg,old,b,s,sa=baseline_build('no_event')
 renderer,*_=tiny();torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer)
 return c,old,model,b,s,sa

def test_all_attention_ffn_adapted_whisper_full_no_event_optimizer():
 c,old,m,b,s,sa=fixture()
 from omni_av_p21.lora import AdaptedLinear
 assert len(m.text_targets)==28 and names(c)==()
 assert not m.event_heads and not any('event_heads.'in k for k in m.state_dict())
 assert {g['name']for g in m.groups()}=={'connector','ltx_lora','whisper','features'}
 for block in m.renderer.transformer_blocks:
  for name in ATTENTION:
   assert all(isinstance(getattr(block,name).get_submodule(p),AdaptedLinear)for p in ['to_q','to_k','to_v','to_out.0'])
  for name in FFN:assert isinstance(getattr(block,name).net[0].proj,AdaptedLinear)and isinstance(getattr(block,name).net[2],AdaptedLinear)
  for name in WHISPER:
   assert all(p.requires_grad for p in getattr(block,name).parameters())
   assert not any(isinstance(x,AdaptedLinear)for x in getattr(block,name).modules())
 ps=[p for g in m.groups()for p in g['params']]
 assert len(ps)==len({id(p)for p in ps});m.assert_ownership()
 assert 'events'not in b and 'hidden_event_layout'not in b

@pytest.mark.parametrize('recompute',[False,True])
def test_original_parameters_and_zero_delta_forward_exact(recompute):
 c,old,m,b,s,sa=fixture()
 for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
 for model in [old,m]:model.train();model.renderer.set_gradient_checkpointing(recompute)
 torch.manual_seed(991);a=forward(old,b,s,sa)
 torch.manual_seed(991);v=forward(m,b,s,sa)
 torch.testing.assert_close(a['av'],v['av'],rtol=0,atol=0)
 for mod in ['video','audio']:torch.testing.assert_close(a['features'][mod],v['features'][mod],rtol=0,atol=0)
 assert not v['event_sums']and not v['event_counts']

@pytest.mark.parametrize('task',['av','full'])
def test_gradients_all_ltx_families_whisper_and_frozen_base(task):
 c,_,m,b,s,sa=fixture();opened(m);m.train();m.renderer.set_gradient_checkpointing(True)
 frozen=[(p,p.detach().clone())for p in m.renderer.parameters()if not p.requires_grad]
 v=forward(m,b,s,sa);loss=v['av']if task=='av'else objective(v,c,{})[0];loss.backward()
 for family in (*ATTENTION,*FFN):
  targets=[x for n,x in zip(m.text_targets,m.text_lora)if n.split('.')[2]==family]
  assert targets and any(p.grad is not None and p.grad.abs().sum()>0 for x in targets for p in x.parameters()),family
 for name in WHISPER:assert any(p.grad is not None and p.grad.abs().sum()>0 for block in m.renderer.transformer_blocks for p in getattr(block,name).parameters())
 torch.optim.AdamW(m.groups(),weight_decay=0.).step()
 assert all(p.grad is None and torch.equal(p,v)for p,v in frozen)
 m.assert_ownership()

def test_feature_only_updates_connector_decoders_and_no_ltx_or_whisper():
 c,_,m,b,s,sa=fixture();o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 for _ in range(3):
  o.zero_grad(set_to_none=True);v=forward(m,b,s,sa);sum(v['features'].values()).backward();o.step()
 for module in [m.connector,m.decoders]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
 for module in [m.text_lora,m.whisper_branches,m.whisper_adapter]:assert all(p.grad is None for p in module.parameters())
 assert not b['condition'].hidden.requires_grad

def test_loss_exactly_av_plus_two_feature_terms():
 c,_,m,b,s,sa=fixture();v=forward(m,b,s,sa);loss,parts,event=objective(v,c,{})
 torch.testing.assert_close(loss,v['av']+.025*v['features']['video']+.025*v['features']['audio'],rtol=0,atol=0)
 assert float(event.detach())==0 and float(parts['clip'].detach())==0 and float(parts['temporal'].detach())==0

def test_checkpoint_restores_next_update_and_rejects_old_schema(tmp_path):
 c,_,m,b,s,sa=fixture();o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity=dict(world=1,scope=m.ltx_lora_targets,arm='expanded_noevent')
 def step():
  o.zero_grad(set_to_none=True);objective(forward(m,b,s,sa),c,{})[0].backward();o.step()
 step();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()]);step();wanted=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 checkpoint.resume(p,m,o,identity);step()
 assert all(torch.equal(v,m.state_dict()[k])for k,v in wanted.items())
 for i,s in adam['state'].items():
  for k,v in s.items():assert torch.equal(v,o.state_dict()['state'][i][k])
 with patch('p2_ltx_all_lora_noevent.checkpoint.VERSION','old'):
  with pytest.raises(ValueError):checkpoint.checked(p)
 with pytest.raises(ValueError):checkpoint.resume(p,m,o,dict(identity,arm='wrong'))

@pytest.mark.parametrize('change',[
 lambda c:c['loss_weights'].update(event=.05),
 lambda c:c['ltx_lora_scope']['attention'].remove('attn1'),
 lambda c:c.update(sample_list='/old/data.csv'),
 lambda c:c['text_lora'].update(rank=32),
 lambda c:c.update(teacher_cache=None),
 lambda c:c.update(component_arm='no_feature'),
])
def test_strict_recipe_rejects_loss_data_or_scope_drift(change):
 c=expected('/tmp/expanded-whisper-feature');validate(c);change(c)
 with pytest.raises(ValueError):validate(c)

def test_reader_never_opens_event_labels():
 from p2_ltx_all_lora_noevent.data import Dataset
 from types import SimpleNamespace
 d=Dataset.__new__(Dataset);d.spec={};d.index={};d.ids=['sample'];d.components=dict(whisper=True,feature=True,event=False)
 teacher=SimpleNamespace(h=SimpleNamespace(protocol={'sample_ids':['sample']}))
 with patch('p2_current_component_ablation.data.open_teacher',return_value=teacher)as t,patch('p2_current_component_ablation.data.open_speaker')as w,patch('p2_current_component_ablation.data.event_readers',side_effect=AssertionError('Event reader must stay closed'))as e:
  d.open_aux();assert t.called and w.called and not e.called

def test_expanded_full_model_shared_initialization_and_av_feature_match():
 from test_p2_ltx_all_lora import fixture as full_fixture
 from test_p2_ltx_events import forward as full_forward
 _,_,full,full_bundle,_,full_strategy,full_sampler=full_fixture()
 c,_,m,b,s,sa=fixture()
 for k,v in m.state_dict().items():assert torch.equal(v,full.state_dict()[k]),k
 torch.manual_seed(818);actual=forward(m,b,s,sa)
 shared=dict(b,events=full_bundle['events'],hidden_event_layout=full_bundle['hidden_event_layout'])
 torch.manual_seed(818);wanted=full(shared,{k:1. for k in shared['events']},torch.Generator().manual_seed(97),full_strategy,full_sampler)
 torch.testing.assert_close(actual['av'],wanted['av'],rtol=0,atol=0)
 for mod in ['video','audio']:torch.testing.assert_close(actual['features'][mod],wanted['features'][mod],rtol=0,atol=0)
