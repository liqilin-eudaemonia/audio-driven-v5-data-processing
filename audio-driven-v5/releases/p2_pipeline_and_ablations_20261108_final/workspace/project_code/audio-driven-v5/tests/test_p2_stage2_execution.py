"""Actual joint update engine, nativeQwen+PEFT+LTX, two-process SUM vs accumulation."""
from pathlib import Path
from unittest.mock import patch
import copy,torch
from test_p2_stage2_joint import joint_fixture
from p2_stage2_joint.training import update


def setup(rank):
 model,g,b,pref,s,sa=joint_fixture('full')
 from p2_listen.contracts import ROOT,read
 g['execution']=read(ROOT/'experiments/p2_current500h_av_whisper31_20261107/config.json')['execution']
 model.qwen.processor.tokenizer.encode=lambda text,**kwargs:([4]if text=='short'else[4,5,6])
 b['planning_target']='short'if rank==0 else'long';b['planning_tokens']=2 if rank==0 else 4
 b['av']['av']['latents']['latents']=b['av']['av']['latents']['latents']+.03*rank
 c=dict(generator=g,planning_weight=.1)
 o=torch.optim.AdamW(model.groups(),weight_decay=0.)
 return model,c,b,pref,s,sa,o


def distributed_worker(rank,path):
 import torch.distributed as dist
 dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(Path(path)/'rendezvous'))
 try:
  m,c,b,pref,s,sa,o=setup(rank);pos={k:1. for k in b['events']}
  for step in range(1,3):
   with patch('p3_listen.features.prefix',return_value=pref):record=update(m,o,c,[b],[rank+13],s,sa,pos,list(pos),step,'cpu',2)
  torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),record=record),Path(path)/f'rank{rank}.pt')
 finally:dist.destroy_process_group()


def test_actual_joint_engine_two_rank_matches_accumulated_global_update(tmp_path):
 import torch.multiprocessing as mp
 mp.spawn(distributed_worker,args=(str(tmp_path),),nprocs=2,join=True)
 m,c,b0,pref,s,sa,o=setup(0);_,_,b1,*_=setup(1);pos={k:1. for k in b0['events']}
 for step in range(1,3):
  with patch('p3_listen.features.prefix',return_value=pref):record=update(m,o,c,[b0,b1],[13,14],s,sa,pos,list(pos),step,'cpu',1)
 for rank in range(2):
  saved=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
  for key,value in m.state_dict().items():torch.testing.assert_close(value,saved['model'][key],rtol=0,atol=0,msg=key)
  for key,state in o.state_dict()['state'].items():
   for name,value in state.items():torch.testing.assert_close(value,saved['optimizer']['state'][key][name],rtol=0,atol=0)
  assert saved['record']['planning_tokens']==record['planning_tokens']==6
  for key in ('av','event','planning_ce','weighted_total'):torch.testing.assert_close(saved['record'][key],record[key],rtol=1e-12,atol=1e-12)


def test_actual_update_canary_probes_both_objectives_on_qwen():
 m,c,b,pref,s,sa,o=setup(0);pos={k:1. for k in b['events']}
 with patch('p3_listen.features.prefix',return_value=pref):r=update(m,o,c,[b],[13],s,sa,pos,list(pos),1,'cpu',1,audit=True)
 probe=r['qwen_objective_gradient_probe'][0]
 assert probe['generation']>0 and probe['weighted_planning']>0
