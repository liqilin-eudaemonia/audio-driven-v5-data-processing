"""Explicit Stage1 model initialization; never import old optimizer or RNG."""
from pathlib import Path
import torch
from p2_listen.contracts import read,sha


def load_generation_state(model,state):
    expected=model.state_dict();extra=set(state)-set(expected)
    # Historical pureAV/Feature/Whisper saved an inert frozen Connector Event
    # head. The Stage2 architecture omits it entirely; no active key may be lost.
    if extra and (model._components['event'] or any(not n.startswith('event_heads.')for n in extra)):
        raise ValueError('Unexpected active Stage1 model keys')
    active={n:v for n,v in state.items()if n in expected}
    if set(active)!=set(expected):raise ValueError('Missing active Stage1 keys')
    for name,value in active.items():
        if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid Stage1 tensor: '+name)
    model.load_state_dict(active,strict=True);model.assert_ownership()
    return dict(active_keys=len(active),ignored_frozen_event_keys=sorted(extra),active_state_exact=True)


def initialize_generator(model,c,verify_checksum=True):
    p=Path(c['assets']['stage1_checkpoint']['path']);marker=read(p.with_suffix('.complete.json'))
    if marker['sha256']!=c['assets']['stage1_checkpoint']['sha256']or verify_checksum and sha(p)!=marker['sha256']:
        raise ValueError('Stage1 checkpoint checksum mismatch')
    if sha(c['parent_config'])!=c['parent_config_sha256']:raise ValueError('Parent configuration changed')
    v=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    if v['step']!=marker['step']or v['identity']['experiment']!=read(c['parent_config']):
        raise ValueError('Stage1 checkpoint is not the selected configuration')
    from .data import data_config
    from p2_listen.features import weights_identity
    data=data_config(c['generator'])
    assets={kind:weights_identity(data,kind)for kind in ('omni','ltx','gemma')}
    if v['identity'].get('assets')!=assets:raise ValueError('Stage1 pretrained base assets changed')
    result=load_generation_state(model,v['model'])
    result.update(parent_step=v['step'],parent_checkpoint_sha256=marker['sha256'],optimizer_reset=True,rng_reset=True,assets=assets)
    return result
