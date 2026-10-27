from pathlib import Path
import json,sys,math
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=Path(__file__).resolve().parent;r=p.parents[1];sys.path.insert(0,str(r/'src'))
from p2_listen.training import checked_checkpoint
from p2_listen.contracts import atomic_json

torch.set_num_threads(4)
def compare(a,b):
 if isinstance(a,torch.Tensor):
  assert a.shape==b.shape and a.dtype==b.dtype
  return int(not torch.equal(a,b)),float((a.double()-b.double()).abs().max())
 if isinstance(a,dict):
  assert a.keys()==b.keys();values=[compare(a[k],b[k]) for k in a]
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b);values=[compare(x,y) for x,y in zip(a,b)]
 else:
  assert a==b,(a,b);return 0,0.
 return sum(x[0] for x in values),max((x[1] for x in values),default=0.)
report={}
one=checked_checkpoint(p/'single/runs/s0_20261026/E1/step_000005.pt')
off=checked_checkpoint(p/'disabled/runs/s0_20261026/E1/step_000005.pt')
report['logging_on_off']={key:dict(zip(['unequal_tensors','max_abs_error'],compare(one[key],off[key]))) for key in ['model','optimizer','rank_rng']}
assert all(v['unequal_tensors']==0 for v in report['logging_on_off'].values())
del off
multi=checked_checkpoint(p/'ddp/runs/s0_20261026/E1/step_000005.pt')
report['single_vs_ddp']={key:dict(zip(['unequal_tensors','max_abs_error'],compare(one[key],multi[key]))) for key in ['model','optimizer']}
assert all(v['unequal_tensors']==0 for v in report['single_vs_ddp'].values())
del one,multi
report['runs']={}
for folder,arm,world in [('single','E0',1),('single','E1',1),('ddp','E1',2)]:
 out=p/folder/f'runs/s0_20261026/{arm}'
 ea=EventAccumulator(str(out/'tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars']
 records=[[json.loads(line) for line in (out/f'train_rank{rank}.jsonl').read_text().splitlines()] for rank in range(world)]
 metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
 assert all([x['step'] for x in rows]==list(range(1,6)) for rows in [*records,metrics])
 assert [x.step for x in ea.Scalars('loss/av')]==list(range(1,6))
 assert all(x.value!=999. for x in ea.Scalars('loss/av'))
 assert 'event_loss/head_shake' not in tags
 assert len(list((out/'tensorboard').glob('events.out.tfevents.*')))==(3 if arm=='E1' else 1)
 for i,row in enumerate(metrics):
  s=row['scalars'];av=sum(rows[i]['av'] for rows in records)/world
  assert math.isclose(s['loss/av'],av,rel_tol=1e-12,abs_tol=1e-12)
  assert math.isclose(s['loss/av_video']+s['loss/av_audio'],av,rel_tol=1e-6)
  assert s['loss/event']==records[0][i]['event']
  assert math.isclose(s['loss/generator_objective'],s['loss/base']+(s['loss/event_weighted'] if arm=='E1' else 0),rel_tol=1e-12)
  for key,value in s.items():assert math.isfinite(value),(key,value)
  assert s['progress/global_batch']==8
  assert s['throughput/samples_per_second']>0
  assert s['memory/peak_allocated_gib_max_rank']>1
  for group,lr in [('connector',3e-5),('events',1e-4),('text_lora',1e-5)]:
   assert math.isclose(s['learning_rate/'+group],lr*row['step']/50,rel_tol=1e-12)
  for tag in ['loss/av','loss/av_video','loss/av_audio','loss/event','grad_norm/main_preclip','throughput/samples_per_second']:
   event=next(x for x in ea.Scalars(tag) if x.step==row['step'])
   assert math.isclose(event.value,s[tag],rel_tol=2e-7,abs_tol=1e-7),(tag,event.value,s[tag])
 report['runs'][folder+'_'+arm]=dict(steps=5,world=world,scalar_tags=tags,rank_zero_only=True,uncommitted_tail_purged=arm=='E1',global_av_verified=True)
assert not (p/'disabled/runs/s0_20261026/E1/tensorboard').exists()
report['passed']=True;atomic_json(p/'gpu_verification.json',report);print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
