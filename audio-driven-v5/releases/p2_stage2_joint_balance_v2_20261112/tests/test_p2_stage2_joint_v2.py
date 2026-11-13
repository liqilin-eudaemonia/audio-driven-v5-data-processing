"""Native small Qwen text model + actual PEFT; no synthetic hidden-gradient stand-in."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from peft import LoraConfig,get_peft_model
from p2_stage2_joint_v2.qwen import OnlineQwen,answer_loss,text_forward
from p2_stage2_joint_v2.model import objective
from p2_stage2_joint_v2 import checkpoint
from p2_listen.training import rng_state

torch.set_num_threads(2)

def build():
 from p2_listen.config import load,activate
 activate(load())
 from test_p2_matrix16 import native_qwen
 torch.manual_seed(123)
 runtime,processor=native_qwen(SimpleNamespace(hidden_layers=[1,2]))
 peft=get_peft_model(runtime.thinker.model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules=['q_proj','v_proj']))
 runtime.thinker.model=peft.get_base_model()
 runtime.thinker.lm_head=torch.nn.Linear(8,32,bias=False)
 online=OnlineQwen(runtime,processor)
 pref=dict(prefix=torch.randn(1,3,8),prefix_valid=torch.ones(1,3,dtype=torch.bool),prefix_positions=torch.arange(3)[None,None].expand(3,1,-1))
 return online,pref

def test_exact_causal_answer_loss_includes_eos_and_only_suffix_predictors():
 q,pref=build()
 with patch('p3_listen.features.prefix',return_value=pref):full=q._full({},'plan',128)
 actual,count=answer_loss(q.runtime.thinker,full,chunk_size=1)
 out=text_forward(q.runtime.thinker,full)
 logits=q.runtime.thinker.lm_head(out.last_hidden_state)
 labels=torch.tensor([[-100,-100,-100,4,5,6,2]])
 expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),reduction='sum')
 torch.testing.assert_close(actual,expected)
 assert count==4
 actual.backward()
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in q.adapters)
 assert all(p.grad is None for p in q.runtime.thinker.lm_head.parameters())

def test_online_selected_hidden_reaches_original_lora_without_training_base():
 q,pref=build();thinker=q.runtime.thinker
 frozen=[(p,p.clone())for p in thinker.parameters()if not p.requires_grad]
 layout=dict(video_coordinates=torch.zeros(1,2,3),audio_coordinates=torch.zeros(1,2,3),video_valid=torch.ones(1,2,dtype=torch.bool),audio_valid=torch.ones(1,2,dtype=torch.bool))
 with patch('p3_listen.features.prefix',return_value=pref):condition=q.condition({},'plan',layout)
 assert condition.hidden.requires_grad
 condition.hidden.square().mean().backward()
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in q.adapters)
 optimizer=torch.optim.AdamW(q.parameters(),lr=1e-6);optimizer.step();q.assert_ownership()
 assert all(p.grad is None and torch.equal(p,v)for p,v in frozen)
 assert set(q.state_dict())=={f'adapters.{i}'for i in range(len(q.adapters))}

def test_global_answer_token_normalization_with_sum_reduction():
 x=torch.nn.Parameter(torch.tensor(.2))
 c=dict(planning_weight=.1,loss_weights=dict(av=1.,event=0.),hidden_event_supervision=dict(clip_weight=0.,temporal_weight=0.))
 def values(n):return dict(av=x.square(),features={},event_sums={},planning_sum=n*(x-1).square(),planning_tokens=n)
 a=objective(values(2),c,{},7,1,2)[0]+objective(values(5),c,{},7,1,2)[0]
 expected=x.square()+.1*(x-1).square()
 torch.testing.assert_close(a,expected)
 torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])

def test_full_adapter_adam_rng_checkpoint_exact_next_update(tmp_path):
 q,pref=build();o=torch.optim.AdamW(q.parameters(),lr=1e-6);identity={'world':1,'adapter_names':q.parameter_names}
 def update():
  o.zero_grad(set_to_none=True)
  with patch('p3_listen.features.prefix',return_value=pref):loss,count=q.planning_loss({},'plan')
  (loss/count).backward();o.step()
 update();p=tmp_path/'step.pt';checkpoint.save(p,q,o,1,identity,[rng_state()]);update()
 expected=copy.deepcopy(q.state_dict());adam=copy.deepcopy(o.state_dict())
 checkpoint.resume(p,q,o,identity);update()
 for k,v in expected.items():assert torch.equal(v,q.state_dict()[k])
 for i,state in adam['state'].items():
  for k,v in state.items():assert torch.equal(v,o.state_dict()['state'][i][k])


def joint_fixture(arm):
 torch.manual_seed(11)
 from test_p2_current_component_ablation import build as gen_build
 from test_p2_ltx_events import build as full_build
 from test_p2_whisper_av import tiny,opened
 from p2_stage2_joint_v2.generator import Model
 from p2_stage2_joint_v2.model import JointModel
 from p2_stage2_joint_v2.initialization import load_generation_state
 if arm=='full':
  c,old,b,_,strategy,sampler=full_build(clip_only=True)
  from test_p2_whisper_event_feature import build as base_build
  _,cfg,*_=base_build()
 else:c,cfg,old,b,strategy,sampler=gen_build(arm)
 f=dict(whisper='whisper'in b,feature=b.get('teacher')is not None,event='events'in b)
 c=copy.deepcopy(c);c['components']=f
 if f['whisper']:opened(old)
 with torch.no_grad():
  for p in old.connector.parameters():
   if p.numel()>1 and not p.any():p.normal_(0,.02)
  for p in old.decoders.parameters():
   if p.numel()>1 and not p.any():p.normal_(0,.02)
 from p2_ltx_all_lora.config import scope
 from p2_ltx_all_lora_components.lora import extend
 c['ltx_lora_scope']=scope()
 if not f['whisper']:c['ltx_lora_scope']['whisper_full']=[]
 extend(old)
 renderer,*_=tiny();gen=Model(c,cfg,renderer)
 receipt=load_generation_state(gen,old.state_dict());assert receipt['active_state_exact']
 q,pref=build();joint=JointModel(gen,q)
 b=dict(b,condition=None,qwen_generation_inputs={},generation_prompt='plan',qwen_planning_inputs={},planning_target='caption',planning_valid=True)
 c['planning_weight']=.0001
 return joint,c,b,pref,strategy,sampler


@pytest.mark.parametrize('arm',['no_whisper','no_feature','no_event','full'])
def test_real_ltx_losses_reach_original_planning_lora(arm):
 joint,c,b,pref,s,sa=joint_fixture(arm)
 with patch('p3_listen.features.prefix',return_value=pref):v=joint(b,{k:1. for k in b.get('events',{})},torch.Generator().manual_seed(97),s,sa)
 losses={'av':v['av'],'planning':v['planning_sum']}
 if v['features']:losses['feature']=sum(v['features'].values())
 if v['event_sums']:losses['event']=sum(v['event_sums'].values())
 for name,loss in losses.items():
  joint.zero_grad(set_to_none=True);loss.backward(retain_graph=True)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for p in joint.qwen.adapters),name
  if name!='planning':
   assert any(p.grad is not None and p.grad.abs().sum()>0 for p in joint.generator.connector.parameters()),name
  else:assert all(p.grad is None for p in joint.generator.parameters())
 joint.assert_ownership()
 groups=joint.groups();ps=[p for g in groups for p in g['params']]
 assert len(ps)==len({id(p)for p in ps})
 assert {id(p)for p in ps}=={id(p)for p in joint.parameters()if p.requires_grad}


def test_complete_joint_checkpoint_exact_next_update(tmp_path):
 joint,c,b,pref,s,sa=joint_fixture('full');o=torch.optim.AdamW(joint.groups(),weight_decay=0.)
 identity={'world':1,'qwen_names':joint.qwen.parameter_names}
 def update():
  o.zero_grad(set_to_none=True)
  torch.manual_seed(333)
  with patch('p3_listen.features.prefix',return_value=pref):v=joint(b,{k:1. for k in b['events']},torch.Generator().manual_seed(97),s,sa)
  objective(v,c,{k:r['known'].sum()for k,r in b['events'].items()},v['planning_tokens'])[0].backward();o.step()
 update();p=tmp_path/'joint.pt';checkpoint.save(p,joint,o,1,identity,[rng_state()]);update()
 expected=copy.deepcopy(joint.state_dict());adam=copy.deepcopy(o.state_dict())
 checkpoint.resume(p,joint,o,identity);update()
 for n,v in expected.items():assert torch.equal(v,joint.state_dict()[n]),n
 for i,state in adam['state'].items():
  for k,v in state.items():assert torch.equal(v,o.state_dict()['state'][i][k])
