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
    optimizer=value.get('optimizer')
    if optimizer is None:raise ValueError('Training resume requires complete Adam state')
    for state in optimizer['state'].values():
        if any(not torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t)):
            raise ValueError('Nonfinite Adam state')
        if 'step' in state and int(state['step'])!=value['step']:
            raise ValueError('Adam step differs from checkpoint')
    return value


def resume(path,model,optimizer,identity,rank=0):
    value=checked(path)
    if value['identity']!=identity:raise ValueError('Combined resume identity mismatch')
    if not 0<=rank<len(value['rank_rng']):raise ValueError('Missing rank RNG')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rank_rng'][rank]);model.assert_ownership()
    return value['step']


def validate_stage(value,c,step,source):
    identity=value['identity']
    if (value['step']!=step or identity['experiment']!=c or identity['source']!=source
            or identity['world']!=28 or len(value['rank_rng'])!=28):
        raise ValueError('Stage checkpoint candidate/source/world mismatch')


def check_initialization(c):
    from .config import WORK
    out=Path(c['output']);current=checked(out/'step_000000.pt')
    baseline_path=WORK/'runs/baseline/step_000000.pt'
    baseline=current if c['loss_search']['candidate']=='baseline' else checked(baseline_path)
    if current['optimizer']['state'] or baseline['optimizer']['state']:raise ValueError('Initialization must have empty Adam history')
    if set(current['model'])!=set(baseline['model'])or any(not torch.equal(t,baseline['model'][k])for k,t in current['model'].items()):raise ValueError('Candidates have different initial model tensors')
    if current['optimizer']['param_groups']!=baseline['optimizer']['param_groups']:raise ValueError('Different initial optimizer groups')
    atomic_json(out/'INITIALIZATION_ACCEPTED.json',dict(success=True,baseline=str(baseline_path),model_tensors=len(current['model']),model_bitwise_equal=True,empty_adam_history=True))
