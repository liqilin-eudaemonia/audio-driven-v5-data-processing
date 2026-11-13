"""Native Whisper AV, exact recovery and final-data scheduling."""
import copy,sys
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import pytest,torch
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_current_av_whisper.config import load,validate,native_view
from p2_current_av_whisper.model import Model,av_objective
from p2_current_av_whisper import checkpoint
from p2_listen.contracts import ROOT,read
from p2_listen.training import rng_state
CONFIG=ROOT/'experiments/p2_current500h_av_whisper31_20261107/config.json'
torch.set_num_threads(2)

def build():
 from test_p2_whisper_av import tiny,features
 from p2_listen.audit import fixture
 from p2_listen.config import load as dc
 from omni_av.ltx_runtime import with_text
 c=copy.deepcopy(load(CONFIG));c.pop('connector');c['whisper'].update(input_dim=12,context_dim=16)
 cfg,f=fixture(dc());cfg=replace(cfg,video_dim=32,audio_dim=32)
 r,s,sa,av,text=tiny();torch.manual_seed(c['training']['seed']);m=Model(c,cfg,r)
 p=s.prepare_training_inputs(with_text(av,text),sa)
 b=dict(condition=f[0],layout=f[1],teacher=None,whisper=features(),text={'positive':text},av=dict(av=av,
  support=dict(video=torch.ones_like(p.video_loss_mask),audio=torch.ones_like(p.audio_loss_mask))))
 return c,cfg,m,b,s,sa

def test_three_native_updates_have_only_av_and_all_three_parameter_groups():
 c,cfg,m,b,s,sa=build();m.renderer.set_gradient_checkpointing(True);o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 frozen=[(p,p.clone())for p in m.renderer.parameters()if not p.requires_grad]
 with patch.object(m.event_heads,'forward',side_effect=AssertionError('Event called')):
  for _ in range(3):
   o.zero_grad(set_to_none=True);v=m(b,s,sa);loss,base,event=av_objective(v,c,{},1,1)
   assert torch.equal(loss,v['av']) and event==0 and set(v)=={'av','av_components'}
   loss.backward();assert torch.isfinite(loss)
   assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.whisper_branches.parameters())
   o.step()
 assert all(p.grad is None and torch.equal(p,old)for p,old in frozen)
 for mod in (m.connector,m.text_lora,m.whisper_adapter):assert any(p.grad is not None and p.grad.abs().sum()>0 for p in mod.parameters())
 assert {g['name']for g in m.groups()}=={'connector','text_lora','whisper'}
 assert not len(m.decoders) and not len(m.qwen_lora)
 assert all(p.grad is None for p in m.event_heads.parameters())
 assert not b['condition'].hidden.requires_grad and not b['whisper'].hidden.requires_grad

def test_unchanged_native_whisper_initialization_and_forward():
 from p2_whisper_av.model import QwenWhisperAV
 from test_p2_whisper_av import tiny
 c,cfg,m,b,s,sa=build();r,*_=tiny();torch.manual_seed(c['training']['seed']);native=QwenWhisperAV(native_view(c),cfg,r)
 assert all(torch.equal(v,native.state_dict()[k])for k,v in m.state_dict().items())
 torch.manual_seed(7);a=m(b,s,sa);torch.manual_seed(7);v=native(b,s,sa)
 torch.testing.assert_close(a['av'],v['av'],rtol=0,atol=0)

def test_exact_adam_rng_next_update_recovery(tmp_path):
 c,cfg,m,b,s,sa=build();o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity={'world':1,'arm':'av_whisper'}
 def update():
  o.zero_grad(set_to_none=True);av_objective(m(b,s,sa),c,{},1,1)[0].backward();o.step()
 update();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()]);update()
 expected=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 assert checkpoint.resume(p,m,o,identity)==1
 update();assert all(torch.equal(v,m.state_dict()[k])for k,v in expected.items())
 for k,state in adam['state'].items():
  for n,v in state.items():assert torch.equal(v,o.state_dict()['state'][k][n])
 with pytest.raises(ValueError):checkpoint.resume(p,m,o,{'world':2})

@pytest.mark.parametrize('key,val',[('teacher_cache','wrong'),('clip_labels','wrong'),('whisper_cache','wrong'),('qwen_hidden_source','sft'),('initialization','joint_resume')])
def test_invalid_sources_rejected(key,val):
 c=load(CONFIG);c[key]=val
 with pytest.raises(ValueError):validate(c)

def test_feature_event_weights_rejected():
 for k in ('event','video_feature','audio_feature'):
  c=load(CONFIG);c['loss_weights'][k]=.025
  with pytest.raises(ValueError):validate(c)

def test_same_samples_and_noise_ordinals_as_av_control():
 from p2_current_av_whisper.schedule import Schedule
 c=load(CONFIG);a=read(ROOT/'experiments/p2_current500h_av_only31_20261107/config.json');s=Schedule(c)
 assert read(c['duration_sampling']['manifest'])['pairs']==read(a['duration_sampling']['manifest'])['pairs']
 assert s.size==284218 and len(s.pairs)==9169*31
 assert c['j_cache']==a['j_cache'] and c['current_cache_index']==a['current_cache_index']

def test_31rank_limits_and_reserve_gpu7(monkeypatch):
 import p2_current500h_av_whisper as cli
 import p2_current_av_whisper.training as training
 c=load(CONFIG);commands=cli.commands(read(Path(c['output'])/'cluster.json'),CONFIG,stop_after=2)
 assert all('P2_AV_WHISPER_STOP_AFTER=2'in cmd for cmd in commands)
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in commands[-1]
 calls=[];monkeypatch.setattr(training,'train',lambda c,r,limit:calls.append(limit))
 monkeypatch.setenv('P2_AV_WHISPER_STOP_AFTER','2');monkeypatch.setattr(sys,'argv',['cli','train','--config',str(CONFIG)])
 cli.main();assert calls==[2]
