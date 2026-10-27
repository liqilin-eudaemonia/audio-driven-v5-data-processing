"""Exact real-model, optimizer and per-rank RNG comparison across startup variants."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
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

def source_allowed(old,new):
 base=ROOT/'src/p2_listen';release=ROOT/'releases/p2_preflight_v1'
 replacements={str(base/name):str(release/'src/p2_listen'/name) for name in ('__init__.py','training.py','config.py')}
 expected={replacements.get(p,p):sha(replacements.get(p,p)) for p in old}
 for p in (release/'src/p2_listen/preflight.py',release/'scripts/p2_listen.py',ROOT/'scripts/p2_listen.py'):expected[str(p)]=sha(p)
 assert new==expected,'Unapproved source delta'
 assert all(sha(p)==h for p,h in old.items()),'Original source changed'

report={}
for left,right,arm,label in [('baseline','fast','E0','baseline_vs_fast_E0'),('baseline','fast','E1','baseline_vs_fast_E1'),('fast','resume','E1','continuous_vs_resumed_E1')]:
 folders=[OUT/k/'runs/s0_20261023'/arm for k in (left,right)]
 a,b=[checked_checkpoint(f/'step_000005.pt') for f in folders]
 ia,ib=a['identity'],b['identity']
 for key in set(ia)|set(ib):
  if key not in ('config','source'):assert ia[key]==ib[key],(label,key)
 ca,cb=dict(ia['config']),dict(ib['config']);ca.pop('output');cb.pop('output');assert ca==cb
 if left=='baseline':source_allowed(ia['source'],ib['source'])
 else:assert ia['source']==ib['source']
 values={k:compare(a[k],b[k],k) for k in ('model','optimizer','rank_rng')}
 for rank in range(6):
  logs=[[json.loads(s) for s in (f/f'train_rank{rank}.jsonl').read_text().splitlines()] for f in folders]
  for ra,rb in zip(*logs):
   for key in ('step','ordinals','base','av','event','event_known_bins','event_loss_per_class','grad_norm'):
    assert ra[key]==rb[key],(label,rank,key)
 report[label]=dict(status='passed',bitwise_equal=True,steps=5,world=6,states=values,
                    initial_parameters_same=True,sample_order_same=True,class_statistics_same=True)
 del a,b
atomic_json(OUT/'gpu_comparison.json',report)
print(json.dumps(report,indent=2))
