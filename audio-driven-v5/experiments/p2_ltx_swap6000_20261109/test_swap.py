from pathlib import Path
import copy,json,sys,hashlib
import pytest
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import swap as S
import gate
S.activate()
import torch
torch.set_num_threads(2)

def metadata(world,out):
 return dict(world=world,experiment=dict(output=str(out),training=dict(global_batch=31),loss='fixed'),source={},data_identity={'fixed':True},statistics={})

@pytest.mark.parametrize('world,new',[(1,31),(31,1)])
def test_identity_changes_only_output_world(world,new,tmp_path):
 parent=metadata(world,tmp_path/'old');c=copy.deepcopy(parent['experiment']);c['output']=str(tmp_path/'new');pre=dict(source={},data_identity={'fixed':True},statistics={})
 target=S.target_identity(parent,c,pre,new);normalized=copy.deepcopy(target);normalized['world']=world;normalized['experiment']['output']=parent['experiment']['output'];assert normalized==parent
 for bad in [dict(c,loss='different'),dict(c,training=dict(global_batch=1)),copy.deepcopy(parent['experiment'])]:
  with pytest.raises(ValueError):S.target_identity(parent,bad,pre,new)
 with pytest.raises(ValueError):S.target_identity(parent,c,dict(pre,source={'drift':'bad'}),new)

@pytest.mark.parametrize('arm,world,new',[('full',31,1),('noevent',1,31)])
def test_real_native_model_and_adam_checkpoint_migration(arm,world,new,tmp_path):
 from p2_listen.training import rng_state
 if arm=='full':
  from test_p2_ltx_all_lora import fixture
  _,_,model,*_=fixture()
 else:
  from test_p2_ltx_all_lora_noevent import fixture
  _,_,model,*_=fixture()
 optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
 for p in model.parameters():
  if p.requires_grad:p.grad=torch.full_like(p,.01)
 optimizer.step();optimizer.zero_grad(set_to_none=True)
 old=tmp_path/'old';newdir=tmp_path/'new';old.mkdir();newdir.mkdir();source=old/'step_000007.pt';destination=newdir/source.name
 identity=metadata(world,old);cp=S.module(arm,'checkpoint');cp.save(source,model,optimizer,7,identity,[rng_state()for _ in range(world)])
 before=S.sha(source);target=copy.deepcopy(identity);target['world']=new;target['experiment']['output']=str(newdir)
 with pytest.raises(ValueError):S.rewrite(arm,source,source,target,identity)
 assert S.sha(source)==before
 proof=S.rewrite(arm,source,destination,target,identity);assert proof['model_and_adam_equal']and S.sha(source)==before
 migrated=cp.checked(destination);assert migrated['step']==7 and len(migrated['rank_rng'])==new
 expected=copy.deepcopy(migrated['model']);expected_adam=copy.deepcopy(migrated['optimizer'])
 with torch.no_grad():
  for p in model.parameters():
   if p.requires_grad:p.add_(1)
 cp.resume(destination,model,optimizer,target,new-1)
 assert S.equal_tree(expected,model.state_dict())and S.equal_tree(expected_adam,optimizer.state_dict())
 assert not torch.cuda.is_initialized()

def test_native_31_sample_accumulation_matches_31_partition_sum_and_adam():
 from p2_ltx_events.heads import objective
 from p2_whisper_av.execution import accumulator
 c=dict(hidden_event_supervision=dict(clip_weight=1.,temporal_weight=0.),loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025))
 counts={'clip/nod':torch.tensor(16),'clip/vocal':torch.tensor(11)}
 def values(p,i):
  torch.manual_seed(42+i);x=torch.randn(2)
  g=torch.Generator().manual_seed(1000042+i);z=torch.randn(2,generator=g)
  return dict(av=(p*x-z).square().mean(),features=dict(video=(p*x).square().mean(),audio=(p-z).square().mean()),event_sums={'clip/nod':(p[0]-i/31).square()if i%2==0 else p.sum()*0,'clip/vocal':(p[1]+i/31).square()if i%3==0 else p.sum()*0})
 p=torch.nn.Parameter(torch.tensor([.2,-.3]));q=torch.nn.Parameter(p.detach().clone());execution=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1)
 a=accumulator([p],31,execution)
 for i in range(31):objective(values(p,i),c,counts,31,1)[0].backward();a.add()
 a.finish(1);total=torch.zeros_like(q,dtype=torch.float64)
 for i in range(31):
  objective(values(q,i),c,counts,1,31)[0].backward();part=accumulator([q],1,execution);part.add();part.finish(1);total.add_(q.grad.double());q.grad=None
 q.grad=total.float();assert torch.equal(p.grad,q.grad)
 one=torch.optim.AdamW([p],lr=1e-4);many=torch.optim.AdamW([q],lr=1e-4);one.step();many.step()
 assert torch.equal(p,q)and S.equal_tree(one.state_dict(),many.state_dict())

def test_exact6000_gate_and_foreign_stop_priority(tmp_path,monkeypatch):
 out=tmp_path/'full';out.mkdir();cfg=out/'config.json';cfg.write_text('{}');gate.write(out/'status.json',dict(step=5999))
 gate.write(tmp_path/'PROTOCOL.json',dict(full_parent=str(out),target_step=6000,owner='owned',full_config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest()))
 monkeypatch.setattr(gate,'WORK',tmp_path)
 monkeypatch.setattr(gate.time,'sleep',lambda _:gate.write(out/'status.json',dict(step=6000)))
 gate.main();assert gate.read(out/'STOP')['owner']=='owned';assert gate.read(tmp_path/'GATE_STOP_REQUESTED.json')['observed_step']==6000
 before=(out/'STOP').read_bytes();assert not gate.request_stop(out/'STOP','foreign');assert(out/'STOP').read_bytes()==before

def test_noevent_cannot_silently_enable_event_loss():
 row=dict(av=.2,event=0.,feature_video=1.,feature_audio=.5,weighted_total=.2375,per_event={},grad_norm={k:1. for k in ['connector','ltx_lora','whisper','features']})
 S.check_rows([row],False)
 with pytest.raises(ValueError):S.check_rows([dict(row,event=.01)],False)
