from pathlib import Path
import torch
from p2_listen.cache import atomic_torch,cpu
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.training import rng_state,restore_rng
from . import VERSION


def save(path,model,optimizer,step,identity,rng):
    atomic_torch(path,dict(schema=VERSION,model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
                           step=step,identity=identity,rank_rng=rng))
    atomic_json(Path(path).with_suffix('.complete.json'),dict(sha256=sha(path),step=step))


def checked(path):
    if sha(path)!=read(Path(path).with_suffix('.complete.json'))['sha256']:raise ValueError('Checkpoint checksum mismatch')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('schema')!=VERSION:raise ValueError('Not a matrix16 checkpoint')
    if not all(torch.isfinite(v).all() for v in value['model'].values()):raise ValueError('Nonfinite checkpoint')
    return value


def initialize_stage2(path,model,c,data_identity,expected_source=None):
    value=checked(path);parent=value['identity'];pc=parent['experiment']
    if c['stage']!=2 or pc['stage']!=1 or pc['id']!=c['parent_experiment'] or pc['text_condition']!=c['text_condition'] or pc['loss']!=c['loss']:
        raise ValueError('Stage2 parent must match text/loss Stage1')
    if parent['data_identity']!=data_identity:raise ValueError('Parent dataset/plan/teacher identity mismatch')
    if pc.get('event_supervision')!=c.get('event_supervision'):
        raise ValueError('Parent event supervision differs')
    if pc.get('connector')!=c.get('connector'):
        raise ValueError('Parent Connector configuration differs')
    from .config import source_identity
    if parent['source']!=(source_identity() if expected_source is None else expected_source):raise ValueError('Parent implementation differs')
    if any(pc[k]!=c[k] for k in ('loss_weights','text_lora','data_config','plan_cache','teacher_cache')):
        raise ValueError('Parent training/condition contract differs')
    state=model.state_dict();expected={k for k in state if not k.startswith('qwen_lora.')}
    if set(value['model'])!=expected:raise ValueError('Parent architecture differs')
    for k,v in value['model'].items():
        if state[k].shape!=v.shape:raise ValueError('Parent tensor shape mismatch')
        state[k]=v
    model.load_state_dict(state,strict=True)
    return dict(checkpoint=str(Path(path).resolve()),sha256=sha(path),step=value['step'],experiment=pc['id'])


def resume(path,model,optimizer,identity,rank=0):
    value=checked(path)
    if value['identity']!=identity:raise ValueError('Strict resume identity mismatch')
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rank_rng'][rank]);return value['step']
