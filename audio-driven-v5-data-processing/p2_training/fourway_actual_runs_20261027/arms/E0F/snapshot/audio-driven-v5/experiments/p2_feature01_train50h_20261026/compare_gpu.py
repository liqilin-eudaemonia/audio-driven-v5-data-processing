"""Exact real-model, optimizer and per-rank RNG comparison across startup variants."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'releases/p2_feature_v1/src'))
from p2_listen.training import checked_checkpoint
from p2_listen.contracts import sha,atomic_json

def compare(a,b,path='',counts=None):
 counts=counts if counts is not None else dict(tensors=0,scalars=0)
 if isinstance(a,torch.Tensor):
  assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape,path
  assert torch.equal(a,b),'Tensor difference: '+path
  counts['tensors']+=1
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'.'+str(k),counts)
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i),counts)
 else:
  assert a==b,(path,a,b);counts['scalars']+=1
 return counts


folders=[OUT/k/'runs/s0_20261023/E0' for k in ('continuous','resume')]
a,b=[checked_checkpoint(f/'step_000005.pt') for f in folders]
for key in set(a['identity'])|set(b['identity']):
 if key!='config':assert a['identity'][key]==b['identity'][key],key
ca,cb=[dict(x['identity']['config']) for x in (a,b)];ca.pop('output');cb.pop('output');assert ca==cb
values={k:compare(a[k],b[k],k) for k in ('model','optimizer','rank_rng')}
for rank in range(24):
 logs=[[json.loads(s) for s in (f/f'train_rank{rank}.jsonl').read_text().splitlines()] for f in folders]
 assert len(logs[0])==len(logs[1])==5
 for ra,rb in zip(*logs):
  for key in ('step','ordinals','base','av','event','event_known_bins','event_loss_per_class','grad_norm','feature_losses','feature_weighted_losses'):
   assert ra[key]==rb[key],(rank,key)
report=dict(status='passed',bitwise_equal=True,steps=5,world=24,states=values)
atomic_json(OUT/'gpu_comparison.json',report);print(json.dumps(report,indent=2))
