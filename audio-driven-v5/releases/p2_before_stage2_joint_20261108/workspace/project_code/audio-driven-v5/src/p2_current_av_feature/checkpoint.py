"""Independent complete AV+Feature states; strict identity forbids joint-run resume."""
from pathlib import Path
import torch
from p2_listen.cache import atomic_torch,cpu
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.training import restore_rng
from . import VERSION


def save(path,model,optimizer,step,identity,rng):
 model.assert_ownership();atomic_torch(path,dict(schema=VERSION,model=cpu(model.state_dict()),
  optimizer=cpu(optimizer.state_dict()),step=step,identity=identity,rank_rng=rng))
 atomic_json(Path(path).with_suffix('.complete.json'),dict(step=step,sha256=sha(path)))


def checked(path):
 marker=read(Path(path).with_suffix('.complete.json'))
 if sha(path)!=marker['sha256']:raise ValueError('AV+Feature checkpoint checksum mismatch')
 v=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 if v.get('schema')!=VERSION or v['step']!=marker['step']:raise ValueError('Not this current500h AV+Feature checkpoint')
 if not all(torch.isfinite(t).all()for t in v['model'].values()):raise ValueError('Nonfinite AV+Feature model')
 if not all(torch.isfinite(t).all()for s in v['optimizer']['state'].values()for t in s.values()if torch.is_tensor(t)):
  raise ValueError('Nonfinite AV+Feature optimizer')
 return v


def resume(path,model,optimizer,identity,rank=0):
 v=checked(path)
 if v['identity']!=identity or len(v['rank_rng'])!=identity['world']:raise ValueError('Strict AV+Feature resume identity mismatch')
 if not 0<=rank<len(v['rank_rng']):raise ValueError('Missing rank RNG')
 model.load_state_dict(v['model'],strict=True);optimizer.load_state_dict(v['optimizer']);restore_rng(v['rank_rng'][rank]);model.assert_ownership()
 return v['step']
