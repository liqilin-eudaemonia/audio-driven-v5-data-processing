"""Real tiny LTX AV updates, exact resume, and no auxiliary branches/readers."""
import copy,sys
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_current_av_only.config import load,validate
from p2_current_av_only.model import Model,av_objective
from p2_current_av_only.schedule import Schedule
from p2_current_av_only import checkpoint
from p2_listen.training import rng_state
from p2_listen.contracts import read
CONFIG=ROOT/'experiments/p2_current500h_av_only_20261107/config.json'
torch.set_num_threads(2)


def build():
 from test_p2_whisper_av import tiny
 from p2_listen.audit import fixture
 from p2_listen.config import load as dc
 from omni_av.ltx_runtime import with_text
 from p2_listen.runtime import av_loss
 from omni_av_p2.ltx import append_conditions
 c=load(CONFIG);c=copy.deepcopy(c);c.pop('connector')
 cfg,f=fixture(dc());cfg=replace(cfg,video_dim=32,audio_dim=32)
 renderer,strategy,sampler,av,text=tiny();torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer)
 prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
 bundle=dict(condition=f[0],teacher=None,text={'positive':text},av=dict(av=av,
  support=dict(video=torch.ones_like(prepared.video_loss_mask),audio=torch.ones_like(prepared.audio_loss_mask))))
 def callback(u,pack,txt):
  p=strategy.prepare_training_inputs(with_text(pack['av'],txt),sampler)
  return av_loss(renderer,append_conditions(p,u),pack['support'],return_components=True)
 return c,cfg,model,bundle,callback


def test_two_real_native_av_updates_never_call_event_or_feature_or_whisper():
 c,cfg,m,b,callback=build();before={k:v.clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.groups(),weight_decay=0.)
 with patch.object(m.event_heads,'forward',side_effect=AssertionError('Event head must never execute')):
  for _ in range(2):
   opt.zero_grad(set_to_none=True);v=m(b,{},None,callback);loss,base,event=av_objective(v,c,{},1,1)
   assert torch.equal(loss,v['av'])and event.item()==0 and v['features']=={}
   loss.backward();assert torch.isfinite(loss)
   assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.connector.parameters())
   assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.text_lora.parameters())
   assert all(p.grad is None and not p.requires_grad for p in m.event_heads.parameters())
   opt.step();m.assert_ownership()
 assert {g['name']for g in m.groups()}=={'connector','text_lora'}
 assert not len(m.decoders)and not len(m.qwen_lora)and not hasattr(m,'whisper_adapter')
 assert all(not hasattr(block,'qwen_attn2')for block in m.renderer.transformer_blocks)
 assert all(torch.equal(v,m.state_dict()[k])for k,v in before.items()if k.startswith('event_heads.'))


def test_identical_initialization_to_original_av_only():
 from p2_j_av_only import AVOnlyModel
 from test_p2_whisper_av import tiny
 c,cfg,m,_,_=build();renderer,*_=tiny();torch.manual_seed(c['training']['seed']);original=AVOnlyModel(c,cfg,renderer)
 assert m.state_dict().keys()==original.state_dict().keys()
 assert all(torch.equal(v,original.state_dict()[k])for k,v in m.state_dict().items())


def test_complete_checkpoint_restores_exact_next_native_update_and_rng(tmp_path):
 c,cfg,m,b,callback=build();opt=torch.optim.AdamW(m.groups(),weight_decay=0.);identity={'world':1,'experiment':c}
 def update(model,optimizer,bundle,cb):
  optimizer.zero_grad(set_to_none=True);av_objective(model(bundle,{},None,cb),c,{},1,1)[0].backward();optimizer.step()
 update(m,opt,b,callback);path=tmp_path/'step_000001.pt';checkpoint.save(path,m,opt,1,identity,[rng_state()])
 update(m,opt,b,callback);expected=copy.deepcopy(m.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
 assert checkpoint.resume(path,m,opt,identity)==1
 update(m,opt,b,callback)
 assert all(torch.equal(v,m.state_dict()[k])for k,v in expected.items())
 for k,state in expected_opt['state'].items():
  for n,v in state.items():assert torch.equal(v,opt.state_dict()['state'][k][n])
 with pytest.raises(ValueError,match='identity'):checkpoint.resume(path,m,opt,dict(identity,world=2))


def test_schedule_exactly_matches_initial_current500h_full_arm_and_all_real_members():
 c=load(CONFIG);s=Schedule(c);old=read(ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106/config.json')
 assert c['duration_sampling']==old['duration_sampling']and s.pairs==list(map(tuple,read(old['duration_sampling']['manifest'])['pairs']))
 real=[(i,o)for i,o in s.pairs if o<s.size]
 assert len(real)==len({i for i,o in real})==len({o for i,o in real})==284218
 assert len(s.pairs)-s.size==6
 for step in [1,17,8882]:
  assert sum((s.indices(s.size,step,32,s.seed,rank,32)for rank in range(32)),[])==s.pairs[(step-1)*32:step*32]


@pytest.mark.parametrize('key,value',[('whisper',{}),('whisper_cache','bad'),('teacher_cache','bad'),('clip_labels','bad'),
 ('initialization','resume_full_joint'),('qwen_hidden_source','planning_sft')])
def test_rejects_unrequested_condition_or_supervision(key,value):
 c=load(CONFIG);c[key]=value
 with pytest.raises(ValueError):validate(c)


def test_rejects_nonzero_auxiliary_weights():
 c=load(CONFIG);c['loss_weights']['event']=.05
 with pytest.raises(ValueError):validate(c)


def test_only_three_readers_open_and_full_train_dev_kept():
 from p2_current_av_only.data import Dataset
 with patch('p2_j_current_whisper_cache.Reader',side_effect=AssertionError('Whisper opened')),\
      patch('p2_j_current_teacher_cache.Reader',side_effect=AssertionError('Teacher opened')),\
      patch('p2_clip_events.labels.LabelReader',side_effect=AssertionError('Labels opened')):
  d=Dataset(load(CONFIG),verify=False)
 assert len(d.rows)==284218 and sum(r['split']=='dev'for r in d.rows)==3158
 assert set(r['split']for r in d.rows)=={'train','dev'}


def test_only_explicit_commands_prepare_and_wrong_resume_is_rejected():
 from p2_current500h_av_only import commands
 c=load(CONFIG);cluster=read(Path(c['output'])/'cluster.json');cmds=commands(cluster,CONFIG)
 assert len(cmds)==4 and all(str(ROOT/'scripts/p2_current500h_av_only.py')in cmd for cmd in cmds)
 assert all('--resume'not in cmd for cmd in cmds)
 with pytest.raises(ValueError):commands(cluster,CONFIG,ROOT/'experiments/p2_ltx_clip_train31_cadence_20261107/step_006000.pt')


def test_cold_training_entry_resolves_dependencies_then_stops_before_cuda(tmp_path):
 import subprocess,os
 code='''import sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd()/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_current_av_only.config import load,source_identity
from p2_current_av_only.data import data_identity
from p2_current_av_only.training import train
from p2_current_av_only import VERSION
from p2_listen.contracts import atomic_json,digest
c=load(sys.argv[1]);c['output']=sys.argv[2]
atomic_json(Path(c['output'])/'preflight.json',dict(schema=VERSION,configuration_sha256=digest(c),data_identity=data_identity(c),source=source_identity()))
train(c,stop_after=1)
'''
 result=subprocess.run([sys.executable,'-c',code,str(CONFIG),str(tmp_path)],cwd=ROOT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=60)
 assert result.returncode!=0 and 'Real LTX training requires GPU'in result.stderr,result.stderr
 assert 'ModuleNotFoundError'not in result.stderr and not (tmp_path/'run.json').exists()


def test_optional24_schedule_has_exact_same_real_sample_noise_membership():
 c=load(ROOT/'experiments/p2_current500h_av_only24_20261107/config.json');s=Schedule(c);reference=Schedule(load(CONFIG))
 assert {o:i for i,o in s.pairs if o<s.size}=={o:i for i,o in reference.pairs if o<reference.size}
 assert len(s.pairs)-s.size==14 and c['training']['steps']==11843


def test_explicit31_layout_reserves_entire_node11_gpu7():
 from p2_current500h_av_only import commands,layout31
 c=load(ROOT/'experiments/p2_current500h_av_only31_20261107/config.json');cluster=read(Path(c['output'])/'cluster.json')
 entries=layout31(cluster);assert [r for n in entries for r in n['ranks']]==list(range(31))
 assert entries[-1]['gpus']==list(range(7))
 cmds=commands(cluster,Path(c['output'])/'config.json')
 assert len(cmds)==4 and 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6'in cmds[-1]
 assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'not in cmds[-1]


def test_31_schedule_preserves_every_real_sample_noise_pair():
 c=load(ROOT/'experiments/p2_current500h_av_only31_20261107/config.json');s=Schedule(c);original=Schedule(load(CONFIG))
 assert {o:i for i,o in s.pairs if o<s.size}=={o:i for i,o in original.pairs if o<original.size}
 assert c['training']['steps']==9169 and len(s.pairs)-s.size==21
