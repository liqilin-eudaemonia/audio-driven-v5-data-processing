import copy
from pathlib import Path
import pytest,torch
from p2_listen.contracts import ROOT
from p2_current_av_event_only.config import load,validate
from p2_current_av_event_only.model import Model,objective
from p2_current_av_event_only import checkpoint
from p2_listen.training import rng_state

OUT=ROOT/'experiments/p2_current_component_ablations31_20261108/av_event'

def build():
 from test_p2_current_component_ablation import build as existing
 from test_p2_whisper_av import tiny
 _,cfg,old,b,s,sa=existing('no_whisper');b['teacher']=None
 c=load(OUT/'config.json');c.pop('connector')
 renderer,*_=tiny();torch.manual_seed(c['training']['seed']);m=Model(c,cfg,renderer)
 for n,v in m.state_dict().items():assert torch.equal(v,old.state_dict()[n]),n
 return c,m,b,s,sa

def forward(m,b,s,sa):
 return m(b,{k:1. for k in b['events']},torch.Generator().manual_seed(97),s,sa)

def test_native_updates_event_path_and_complete_exact_resume(tmp_path):
 c,m,b,s,sa=build();frozen=[(p,p.clone())for p in m.renderer.parameters()if not p.requires_grad]
 assert len(m.decoders)==0 and not hasattr(m,'whisper_adapter')
 assert {g['name']for g in m.groups()}=={'connector','text_lora','events'}
 o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity={'world':1,'arm':'av_event'}
 def update():
  o.zero_grad(set_to_none=True);v=forward(m,b,s,sa);counts={k:r['known'].sum()for k,r in b['events'].items()}
  loss,parts,event=objective(v,c,counts)
  assert not v['features']and parts['temporal']==0
  torch.testing.assert_close(loss,v['av']+.05*event,rtol=0,atol=0)
  loss.backward()
  for g in o.param_groups:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in g['params'])
  o.step();m.assert_ownership()
 update();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()]);update()
 expected=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 assert checkpoint.resume(p,m,o,identity)==1
 update()
 for n,v in expected.items():assert torch.equal(v,m.state_dict()[n])
 for i,state in adam['state'].items():
  for n,v in state.items():assert torch.equal(v,o.state_dict()['state'][i][n])
 assert all(p.grad is None and torch.equal(p,v)for p,v in frozen)
 o.zero_grad(set_to_none=True);sum(forward(m,b,s,sa)['event_sums'].values()).backward()
 for module in (m.connector,m.text_lora,m.event_heads):assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())

def test_all_recompute_and_loss_gradients_identical():
 from p2_listen.performance import configure_renderer
 c,m,b,s,sa=build();results=[]
 for policy in ('none','all'):
  configure_renderer(m.renderer,policy);m.train();m.zero_grad(set_to_none=True);torch.manual_seed(31)
  v=forward(m,b,s,sa);loss=objective(v,c,{k:r['known'].sum()for k,r in b['events'].items()})[0];loss.backward()
  results.append((loss.detach().clone(),{n:p.grad.clone()if p.grad is not None else None for n,p in m.named_parameters()}))
 assert torch.equal(results[0][0],results[1][0])
 for n,g in results[0][1].items():
  if g is None:assert results[1][1][n]is None
  else:assert torch.equal(g,results[1][1][n]),n

def test_disabled_inputs_and_config_rejected():
 c,m,b,s,sa=build()
 for key in ('teacher','whisper'):
  wrong=dict(b,**{key:object()})
  with pytest.raises(ValueError):forward(m,wrong,s,sa)
 c=load(OUT/'config.json')
 for key in ('video_feature','audio_feature'):
  wrong=copy.deepcopy(c);wrong['loss_weights'][key]=.025
  with pytest.raises(ValueError):validate(wrong)

def test_optional_readers_and_launcher_only_event():
 from unittest.mock import patch
 from p2_current_av_event_only.data import Dataset
 import p2_current500h_av_event_only as cli
 from p2_listen.contracts import read
 d=Dataset.__new__(Dataset);d.spec={};d.index={};d.components={'whisper':False,'feature':False,'event':True};d.ids=[]
 with patch('p2_current_av_event_only.data.open_teacher')as t,patch('p2_current_av_event_only.data.open_speaker')as w,patch('p2_current_av_event_only.data.event_readers')as e:
  d.open_aux();assert e.called and not t.called and not w.called
 commands=cli.commands(read(OUT/'cluster.json'),OUT/'config.json',stop_after=6000)
 assert len(commands)==4 and all('P2_COMPONENT_STOP_AFTER=6000'in cmd for cmd in commands)
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in commands[-1]
