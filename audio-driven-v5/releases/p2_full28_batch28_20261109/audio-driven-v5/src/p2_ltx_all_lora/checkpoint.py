"""Independent expanded-LoRA complete checkpoints; old checkpoints are rejected."""
from pathlib import Path
import torch
from p2_listen.cache import atomic_torch,cpu
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.training import restore_rng
from . import VERSION


def save(path,model,optimizer,step,identity,rng):
    model.assert_ownership()
    atomic_torch(path,dict(schema=VERSION,model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
                          step=step,identity=identity,rank_rng=rng))
    atomic_json(Path(path).with_suffix('.complete.json'),dict(sha256=sha(path),step=step))


def checked(path):
    marker=read(Path(path).with_suffix('.complete.json'))
    if sha(path)!=marker['sha256']:raise ValueError('Combined Whisper checkpoint checksum mismatch')
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if value.get('schema')!=VERSION or value['step']!=marker['step']:
        raise ValueError('Not a complete combined Whisper checkpoint')
    if not all(torch.isfinite(t).all() for t in value['model'].values()):
        raise ValueError('Nonfinite combined checkpoint tensor')
    return value


def resume(path,model,optimizer,identity,rank=0):
    value=checked(path)
    if value['identity']!=identity:raise ValueError('Combined resume identity mismatch')
    if not 0<=rank<len(value['rank_rng']):raise ValueError('Missing rank RNG')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rank_rng'][rank]);model.assert_ownership()
    return value['step']
