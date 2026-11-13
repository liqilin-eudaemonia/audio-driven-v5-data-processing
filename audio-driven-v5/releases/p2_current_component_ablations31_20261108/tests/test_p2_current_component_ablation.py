"""Real tiny native LTX: branch removal, gradient paths, initialization and resume."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import pytest,torch
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import ROOT,read
from p2_current_component_ablation import ARMS
from p2_current_component_ablation.config import load,validate
from p2_current_component_ablation.model import Model,objective
from p2_current_component_ablation import checkpoint
from p2_listen.training import rng_state
BASE=ROOT/'experiments/p2_current_component_ablations31_20261108'
torch.set_num_threads(2)

def build(arm):
 from test_p2_whisper_event_feature import build as native_build
 from test_p2_whisper_av import tiny
 from p2_ltx_events.layout import make_layout
 old,cfg,_,bundle,prepared,strategy,sampler=native_build()
 c=load(BASE/arm/'config.json');c.pop('connector')
 if ARMS[arm]['whisper']:c['whisper']=old['whisper']
 else:bundle.pop('whisper')
 if not ARMS[arm]['feature']:bundle['teacher']=None
 if ARMS[arm]['event']:
  bundle['events']={'clip/'+k:v for k,v in bundle['events'].items()}
  bundle['hidden_event_layout']=make_layout(bundle['av']['av'],bundle['av']['support'],.36)
 else:bundle.pop('events')
 renderer,*_=tiny();torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer)
 return c,cfg,model,bundle,strategy,sampler

def forward(m,b,s,sa):return m(b,{k:1. for k in b.get('events',{})},torch.Generator().manual_seed(97),s,sa)

@pytest.mark.parametrize('arm',ARMS)
def test_two_native_updates_correct_losses_and_only_active_parameters(arm):
 c,cfg,m,b,s,sa=build(arm);f=ARMS[arm]
 frozen=[(p,p.clone())for p in m.renderer.parameters()if not p.requires_grad]
 groups={'connector','text_lora'}|({'whisper'}if f['whisper']else set())|({'features'}if f['feature']else set())|({'events'}if f['event']else set())
 assert {g['name']for g in m.groups()}==groups
 optimizer=torch.optim.AdamW(m.groups(),weight_decay=0.);seen=set()
 for _ in range(2):
  optimizer.zero_grad(set_to_none=True);v=forward(m,b,s,sa)
  counts={k:x['known'].sum()for k,x in b.get('events',{}).items()}
  loss,parts,event=objective(v,c,counts)
  expected=v['av']+c['loss_weights']['event']*event+sum(c['loss_weights'][k+'_feature']*x for k,x in v['features'].items())
  torch.testing.assert_close(loss,expected);assert torch.isfinite(loss)
  assert bool(v['features'])==f['feature']and bool(v['event_sums'])==f['event']and parts['temporal']==0
  loss.backward()
  for g in optimizer.param_groups:
   if any(p.grad is not None and p.grad.abs().sum()>0 for p in g['params']):seen.add(g['name'])
  optimizer.step();m.assert_ownership()
 assert seen==groups and all(p.grad is None and torch.equal(p,v)for p,v in frozen)
 assert not b['condition'].hidden.requires_grad
 if not f['event']:assert len(m.event_heads)==0
 if not f['feature']:assert len(m.decoders)==0
 if not f['whisper']:assert not hasattr(m,'whisper_adapter')

@pytest.mark.parametrize('arm',['no_whisper','no_feature'])
def test_event_alone_reaches_ltx_and_connector_not_feature_decoder(arm):
 from test_p2_whisper_av import opened
 c,cfg,m,b,s,sa=build(arm)
 if ARMS[arm]['whisper']:opened(m)
 sum(forward(m,b,s,sa)['event_sums'].values()).backward()
 for module in [m.connector,m.text_lora,m.event_heads]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
 assert all(p.grad is None for p in m.decoders.parameters())
 if ARMS[arm]['whisper']:
  for module in [m.whisper_adapter,m.whisper_branches]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())

@pytest.mark.parametrize('arm',['no_whisper','no_event'])
def test_feature_alone_reaches_connector_and_decoders(arm):
 c,cfg,m,b,s,sa=build(arm);o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 # The native decoder output starts at zero. Feature-only updates first open
 # that output, then propagate into the Connector, exactly as the existing arm.
 for _ in range(3):
  o.zero_grad(set_to_none=True);sum(forward(m,b,s,sa)['features'].values()).backward();o.step()
 for module in [m.connector,m.decoders]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
 assert all(p.grad is None for p in m.text_lora.parameters())

@pytest.mark.parametrize('arm',ARMS)
def test_shared_initialization_matches_existing_full_model(arm):
 from test_p2_ltx_events import build as full_build
 c,cfg,m,b,s,sa=build(arm);_,full,*_=full_build(clip_only=True)
 expected=full.state_dict()
 for name,v in m.state_dict().items():assert torch.equal(v,expected[name]),name

@pytest.mark.parametrize('arm',ARMS)
def test_native_av_noise_and_forward_match_full_model_at_shared_initialization(arm):
 from test_p2_ltx_events import build as full_build
 c,cfg,m,b,s,sa=build(arm);_,full,fb,_,fs,fsa=full_build(clip_only=True)
 # Extra Whisper residuals are zero at initialization. Removing an auxiliary
 # must not consume different AV diffusion RNG or change the native AV target.
 full_bundle=dict(b,teacher=fb['teacher'],whisper=fb['whisper'],events=fb['events'],hidden_event_layout=fb['hidden_event_layout'])
 torch.manual_seed(119);actual=forward(m,b,s,sa)
 torch.manual_seed(119);expected=forward(full,full_bundle,fs,fsa)
 torch.testing.assert_close(actual['av'],expected['av'],rtol=0,atol=0)
 for mod in ('video','audio'):torch.testing.assert_close(actual['av_components'][mod],expected['av_components'][mod],rtol=0,atol=0)

@pytest.mark.parametrize('arm',ARMS)
def test_strict_complete_checkpoint_next_update_and_cross_arm_rejection(arm,tmp_path):
 c,cfg,m,b,s,sa=build(arm);o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity={'world':1,'arm':arm}
 def update():
  o.zero_grad(set_to_none=True);v=forward(m,b,s,sa);objective(v,c,{k:x['known'].sum()for k,x in b.get('events',{}).items()})[0].backward();o.step()
 update();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()]);update();expected=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 assert checkpoint.resume(p,m,o,identity)==1
 update();assert all(torch.equal(v,m.state_dict()[k])for k,v in expected.items())
 for i,state in adam['state'].items():
  for k,v in state.items():assert torch.equal(v,o.state_dict()['state'][i][k])
 with pytest.raises(ValueError):checkpoint.resume(p,m,o,{'world':1,'arm':'wrong'})

@pytest.mark.parametrize('arm',ARMS)
def test_optional_readers_are_only_opened_when_enabled(arm):
 from p2_current_component_ablation.data import Dataset
 d=Dataset.__new__(Dataset);d.spec={};d.index={};d.components=ARMS[arm];d.ids=['sample']
 teacher=SimpleNamespace(h=SimpleNamespace(protocol={'sample_ids':['sample']}))
 with patch('p2_current_component_ablation.data.open_teacher',return_value=teacher)as t,patch('p2_current_component_ablation.data.open_speaker')as w,patch('p2_current_component_ablation.data.event_readers')as e:
  d.open_aux()
  assert t.called==ARMS[arm]['feature']and w.called==ARMS[arm]['whisper']and e.called==ARMS[arm]['event']

@pytest.mark.parametrize('arm',ARMS)
def test_configs_share_final_data_schedule_and_reject_mutations(arm):
 c=load(BASE/arm/'config.json');ref=read(c['component_reference'])
 assert c['duration_sampling']==ref['duration_sampling']and c['sample_list']==ref['sample_list']and c['training']==ref['training']
 for change in [lambda x:x.update(initialization='joint_checkpoint'),lambda x:x['loss_weights'].update(event=.5),lambda x:x.update(j_cache='/old/train_only'),lambda x:x['training'].update(save_every=1000)]:
  wrong=copy.deepcopy(c);change(wrong)
  with pytest.raises(ValueError):validate(wrong)

@pytest.mark.parametrize('arm',ARMS)
def test_31rank_launcher_reserves_gpu7_and_propagates_stop_limit(arm):
 import p2_current500h_component_ablations as cli
 p=BASE/arm/'config.json';c=load(p);cluster=read(p.parent/'cluster.json');commands=cli.commands(cluster,p,stop_after=2)
 assert len(commands)==4 and all('P2_COMPONENT_STOP_AFTER=2'in cmd for cmd in commands)
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in commands[-1]
 assert all(str(cli.SELF)in cmd for cmd in commands)
 with pytest.raises(ValueError):cli.commands(cluster,p,resume=BASE/'other/step.pt')

@pytest.mark.parametrize('arm',ARMS)
def test_rank_entry_uses_stop_limit_from_launcher_environment(arm,monkeypatch):
 import sys,p2_current500h_component_ablations as cli
 import p2_current_component_ablation.training as training
 calls=[];monkeypatch.setenv('P2_COMPONENT_STOP_AFTER','2')
 monkeypatch.setattr(training,'train',lambda c,resume,stop:calls.append((c['component_arm'],stop)))
 monkeypatch.setattr(sys,'argv',['cli','train','--config',str(BASE/arm/'config.json')]);cli.main()
 assert calls==[(arm,2)]

@pytest.mark.parametrize('arm',ARMS)
def test_disabled_input_is_rejected(arm):
 c,cfg,m,b,s,sa=build(arm)
 if arm=='no_whisper':b['whisper']=object()
 elif arm=='no_feature':b['teacher']={}
 else:b['events']={}
 with pytest.raises(ValueError):forward(m,b,s,sa)

def test_explicit_launch_refuses_busy_resources_without_launching(tmp_path,monkeypatch):
 import sys,p2_current500h_component_ablations as cli
 c=load(BASE/'no_feature/config.json')
 with patch.object(cli,'launcher',return_value=SimpleNamespace(selected_resources=lambda cluster:{'selected_all_idle':False})),patch.object(cli,'atomic_json'),patch('listen_cluster.launch',side_effect=AssertionError('Must not launch')):
  monkeypatch.setattr(sys,'argv',['cli','launch','--config',str(BASE/'no_feature/config.json'),'--logs',str(tmp_path)])
  with pytest.raises(RuntimeError,match='busy'):cli.main()

def _distributed_values(x,arm,rank):
 from p2_ltx_events.heads import event_sums
 f=ARMS[arm];labels={}
 if f['event']:
  labels={'clip/a':dict(y=torch.tensor([float(rank)]),known=torch.tensor([True]),weight=torch.tensor([1. if rank else .1])),
          'clip/b':dict(y=torch.tensor([0.]),known=torch.tensor([rank==0]),weight=torch.tensor([.1]))}
 sums,counts=event_sums({k:x.sum()[None]*(i+1)for i,k in enumerate(labels)},labels,{k:1.5 for k in labels})if labels else({},{})
 v=dict(av=(x-(rank+1)).square().mean(),features={m:(x*(rank+2)).square().mean()for m in ('video','audio')}if f['feature']else{},event_sums=sums)
 c=dict(loss_weights=dict(av=1.,event=.05 if f['event']else 0.,video_feature=.025 if f['feature']else 0.,audio_feature=.025 if f['feature']else 0.),hidden_event_supervision=dict(clip_weight=1. if f['event']else 0.,temporal_weight=0.))
 return v,labels,c

def _distributed_worker(rank,arm,folder):
 import torch.distributed as dist
 from p2_current_component_ablation.training import _event_counts
 from p2_whisper_av.execution import StreamingFP64Reduction
 torch.set_num_threads(1);dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(Path(folder)/'rendezvous'))
 try:
  x=torch.nn.Parameter(torch.tensor([.2,-.4]));o=torch.optim.AdamW([x],lr=.001,weight_decay=0.)
  for _ in range(3):
   o.zero_grad(set_to_none=True);v,labels,c=_distributed_values(x,arm,rank)
   counts=_event_counts([{'events':labels}],list(labels),'cpu',2)
   objective(v,c,counts,1,2)[0].backward();acc=StreamingFP64Reduction([x],1);acc.add();acc.finish(2);o.step()
  torch.save(dict(parameter=x.detach(),optimizer=o.state_dict()),Path(folder)/f'rank{rank}.pt')
 finally:dist.destroy_process_group()

@pytest.mark.parametrize('arm',ARMS)
def test_real_two_process_sum_reduction_and_adam_match_reference(arm,tmp_path):
 import torch.multiprocessing as mp
 from p2_current_component_ablation.training import _event_counts
 mp.spawn(_distributed_worker,args=(arm,str(tmp_path)),nprocs=2,join=True)
 x=torch.nn.Parameter(torch.tensor([.2,-.4]));o=torch.optim.AdamW([x],lr=.001,weight_decay=0.)
 for _ in range(3):
  items=[_distributed_values(x,arm,r)for r in range(2)];counts=_event_counts([{'events':labels}for v,labels,c in items],list(items[0][1]),'cpu',1);grads=[]
  for v,labels,c in items:
   o.zero_grad(set_to_none=True);objective(v,c,counts,1,2)[0].backward();grads.append(x.grad.double().clone())
  x.grad=sum(grads).float();o.step()
 for rank in range(2):
  saved=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
  torch.testing.assert_close(saved['parameter'],x,rtol=0,atol=0)
  for k,v in o.state_dict()['state'][0].items():torch.testing.assert_close(saved['optimizer']['state'][0][k],v,rtol=0,atol=0)
