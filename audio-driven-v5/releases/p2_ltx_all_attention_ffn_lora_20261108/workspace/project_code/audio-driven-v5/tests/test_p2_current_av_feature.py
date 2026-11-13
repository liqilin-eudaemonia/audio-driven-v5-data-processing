"""Native AV+Feature, exact initialization/resume and final selection contracts."""
import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import pytest,torch
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_current_av_feature.config import load,validate
from p2_current_av_feature.model import Model,av_objective
from p2_current_av_feature import checkpoint
from p2_listen.contracts import ROOT,read
from p2_listen.training import rng_state
CONFIG=ROOT/'experiments/p2_current500h_av_feature31_20261107/config.json'
torch.set_num_threads(2)

def build():
 from test_p2_whisper_av import tiny
 from p2_listen.audit import fixture
 from p2_listen.config import load as dc
 from omni_av.ltx_runtime import with_text
 from p2_listen.runtime import av_loss
 from omni_av_p2.ltx import append_conditions
 c=copy.deepcopy(load(CONFIG));c.pop('connector')
 cfg,f=fixture(dc());cfg=replace(cfg,video_dim=32,audio_dim=32)
 r,s,sa,av,text=tiny();torch.manual_seed(c['training']['seed']);m=Model(c,cfg,r)
 prepared=s.prepare_training_inputs(with_text(av,text),sa)
 b=dict(condition=f[0],layout=f[1],teacher=f[2],text={'positive':text},av=dict(av=av,
  support=dict(video=torch.ones_like(prepared.video_loss_mask),audio=torch.ones_like(prepared.audio_loss_mask))))
 def cb(u,pack,txt):
  p=s.prepare_training_inputs(with_text(pack['av'],txt),sa)
  return av_loss(r,append_conditions(p,u),pack['support'],return_components=True)
 return c,cfg,m,b,cb

def test_three_native_updates_no_event_and_feature_alone_reaches_connector():
 c,cfg,m,b,cb=build();o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 frozen=[(p,p.clone())for p in m.renderer.parameters()if not p.requires_grad]
 with patch.object(m.event_heads,'forward',side_effect=AssertionError('Event executed')):
  for _ in range(3):
   o.zero_grad(set_to_none=True);v=m(b,{},None,cb)
   loss,base,event=av_objective(v,c,{},1,1)
   torch.testing.assert_close(loss,v['av']+.025*(v['features']['video']+v['features']['audio']))
   assert event==0
   loss.backward();o.step()
  o.zero_grad(set_to_none=True);v=m(b,{},None,cb);sum(v['features'].values()).backward()
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.connector.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.decoders.parameters())
 assert all(p.grad is None for p in m.text_lora.parameters())
 assert all(p.grad is None and torch.equal(p,old)for p,old in frozen)
 assert all(p.grad is None and not p.requires_grad for p in m.event_heads.parameters())
 assert not b['condition'].hidden.requires_grad
 assert {g['name']for g in m.groups()}=={'connector','text_lora','features'}
 assert not hasattr(m,'whisper_adapter')

def test_native_original_feature_initialization():
 from p2_matrix16.model import MatrixModel
 from test_p2_whisper_av import tiny
 c,cfg,m,b,cb=build();old=copy.deepcopy(c);old['loss_weights']['event']=.05
 r,*_=tiny();torch.manual_seed(c['training']['seed']);native=MatrixModel(old,cfg,r)
 assert all(torch.equal(v,native.state_dict()[k])for k,v in m.state_dict().items())

def test_complete_resume_exact_next_update(tmp_path):
 c,cfg,m,b,cb=build();o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity={'world':1,'arm':'av_feature'}
 def update():
  o.zero_grad(set_to_none=True);av_objective(m(b,{},None,cb),c,{},1,1)[0].backward();o.step()
 update();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()])
 update();expected=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 assert checkpoint.resume(p,m,o,identity)==1
 update()
 assert all(torch.equal(v,m.state_dict()[k])for k,v in expected.items())
 for k,state in adam['state'].items():
  for n,v in state.items():assert torch.equal(v,o.state_dict()['state'][k][n])
 with pytest.raises(ValueError):checkpoint.resume(p,m,o,{'world':2})

@pytest.mark.parametrize('key,val',[('whisper',{}),('clip_labels','bad'),('teacher_cache',None),('initialization','joint_checkpoint'),('qwen_hidden_source','sft')])
def test_reject_incompatible_sources(key,val):
 c=load(CONFIG);c[key]=val
 with pytest.raises(ValueError):validate(c)

def test_same_av31_schedule_and_full_membership():
 from p2_current_av_feature.schedule import Schedule
 c=load(CONFIG);a=read(ROOT/'experiments/p2_current500h_av_only31_20261107/config.json');s=Schedule(c)
 assert read(c['duration_sampling']['manifest'])['pairs']==read(a['duration_sampling']['manifest'])['pairs']
 assert len(s.pairs)==9169*31 and s.size==284218
 assert c['j_cache']==a['j_cache'] and c['current_cache_index']==a['current_cache_index']
 assert c['event_head_training'] is False

def test_distributed_normalization():
 x=torch.tensor(2.,requires_grad=True);c=load(CONFIG)
 v={'av':x.square(),'features':{'video':x.square(),'audio':x.square()},'event_sums':{}}
 av_objective(v,c,{},2,31)[0].backward()
 torch.testing.assert_close(x.grad,torch.tensor(4*1.05/62))

def test_31rank_launch_preserves_stop_limit_and_reserves_gpu7():
 import p2_current500h_av_feature as cli
 c=load(CONFIG);cluster=read(Path(c['output'])/'cluster.json')
 commands=cli.commands(cluster,CONFIG,stop_after=2)
 assert len(commands)==4
 assert all('P2_AV_FEATURE_STOP_AFTER=2'in command for command in commands)
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in commands[-1]
 assert all('P2_AV_FEATURE_STOP_AFTER='in command for command in cli.commands(cluster,CONFIG))
 with pytest.raises(ValueError):cli.commands(cluster,CONFIG,resume='/wrong/step.pt')
 with pytest.raises(ValueError):cli.commands(cluster,CONFIG,stop_after=0)

def test_rank_entry_consumes_stop_limit(monkeypatch):
 import sys,p2_current500h_av_feature as cli
 import p2_current_av_feature.training as training
 calls=[]
 monkeypatch.setattr(training,'train',lambda c,resume,stop_after:calls.append(stop_after))
 monkeypatch.setenv('P2_AV_FEATURE_STOP_AFTER','2')
 monkeypatch.setattr(sys,'argv',['cli','train','--config',str(CONFIG)])
 cli.main()
 assert calls==[2]
