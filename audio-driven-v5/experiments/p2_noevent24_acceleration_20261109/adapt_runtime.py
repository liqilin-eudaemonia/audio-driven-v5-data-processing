"""Explicit, policy-bound acceleration of the unchanged batch31 objective."""
from pathlib import Path
import importlib.util,sys,json,torch,functools
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime as original
_CACHE={}
@functools.lru_cache(maxsize=1)
def policy():return json.loads((W/'ADOPTED_POLICY.json').read_text())
def source():
 from p2_listen.contracts import sha
 result=original.source_identity('noevent')
 for name in ['adapt_runtime.py','worker.py','training_adopted.py','ADOPTED_POLICY.json']:result[str(W/name)]=sha(W/name)
 return result

def norms(groups):
 out={}
 for g in groups:
  values=[torch.linalg.vector_norm(p.grad,dtype=torch.float64).square()for p in g['params']if p.grad is not None]
  out[g['name']]=float(torch.stack(values).sum().sqrt())if values else 0.
 return out

def flags(optimizer):
 if policy()['fused']:
  for g in optimizer.param_groups:g.update(fused=True,foreach=False)
  for parameter,state in optimizer.state.items():state['step']=state['step'].to(parameter.device)

def accumulator(parameters,micro_count,execution):
 from p2_whisper_av.execution import accumulator as native
 parameters=list(parameters);p=policy()
 if not p['reuse']:return native(parameters,micro_count,execution)
 key=(tuple(id(v)for v in parameters),micro_count,execution['gradient_bucket_mb'])
 if key not in _CACHE:_CACHE[key]=native(parameters,micro_count,execution)
 return _CACHE[key]

def load():
 package='p2_ltx_all_lora_noevent';spec=importlib.util.spec_from_file_location(package+'._accelerated24',W/'training_adopted.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 old=original.load_training('noevent');m.Schedule=old.Schedule;m.source_identity=source;m.accumulator=accumulator;m.fast_norms=norms;m.optimizer_flags=flags
 def canary(**kw):
  original.longest_canary(training=m,arm='noevent',**kw)
  flags(kw['optimizer'])
 m.split_canary=canary
 return m
