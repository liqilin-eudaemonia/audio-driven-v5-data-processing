"""Atomic model + per-rank optimizer shards. Stage transitions are explicit, sequential."""
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from omni_av_p2.checkpoint import atomic_save, atomic_json_new, capture_rng, restore_rng
from .checkpoint import validate_state
from .residual_config import VERSION


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            value.update(b)
    return value.hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def local_optimizer(optimizer):
    return getattr(optimizer, 'optim', optimizer)


def optimizer_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in local_optimizer(optimizer).param_groups]


def save(path, model, optimizer, step, identity, rank=0, world=1):
    path = Path(path)
    model.assert_parameter_ownership()
    local = path.with_name(path.stem + f'.rank{rank:03d}.pt')
    atomic_save(local, dict(optimizer=cpu(local_optimizer(optimizer).state_dict()),
                            names=optimizer_names(model, optimizer), rng=capture_rng(), rank=rank, world=world))
    record = dict(file=local.name, sha256=sha(local), rank=rank)
    records = [None] * world
    if world > 1:
        dist.all_gather_object(records, record)
    else:
        records = [record]
    if rank == 0:
        state = cpu(model.state_dict())
        # Avoid a second full-model CPU clone. Serialization is synchronous.
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise FloatingPointError('Nonfinite checkpoint')
        atomic_save(path, dict(version=VERSION, architecture=model.architecture(), identity=identity,
                              step=step, stage=model.stage, model=state))
        atomic_json_new(path.with_suffix('.complete.json'), dict(step=step, stage=model.stage,
            world=world, sha256=sha(path), optimizer_shards=records))
    if world > 1:
        dist.barrier()


def read(path):
    path = Path(path)
    marker = json.loads(path.with_suffix('.complete.json').read_text())
    if not {'sha256','stage','step','world','optimizer_shards'} <= marker.keys():
        raise ValueError('Not a residual completion marker; legacy concat checkpoints are incompatible')
    if sha(path) != marker['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('version') != VERSION:
        raise ValueError('Not a residual P2.1 checkpoint; concatenation checkpoints are incompatible')
    if saved['step'] != marker['step'] or saved['stage'] != marker['stage']:
        raise ValueError('Checkpoint marker mismatch')
    return saved, marker


def resume(path, model, optimizer, identity, rank=0, world=1, saved_bundle=None):
    saved, marker = saved_bundle if saved_bundle is not None else read(path)
    if (saved.get('deployment_only') or saved['architecture'] != model.architecture()
            or saved['identity'] != identity or marker['world'] != world):
        raise ValueError('Strict resume requires same stage, architecture, data and topology')
    validate_state(model, saved['model'])
    model.load_state_dict(saved['model'], strict=True)
    shard = marker['optimizer_shards'][rank]
    p = Path(path).with_name(shard['file'])
    if sha(p) != shard['sha256']:
        raise ValueError('Optimizer shard checksum mismatch')
    state = torch.load(p, map_location='cpu', weights_only=True)
    if state['rank'] != rank or state['world'] != world or state['names'] != optimizer_names(model, optimizer):
        raise ValueError('Optimizer partition changed')
    local_optimizer(optimizer).load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return saved


def initialize_previous(path, model, identity, saved_bundle=None):
    saved, _ = saved_bundle if saved_bundle is not None else read(path)
    if saved.get('deployment_only') or saved['stage'] != model.stage - 1:
        raise ValueError('Initialization must use full checkpoint from immediately previous stage')
    for key in ('model', 'lora', 'residual'):
        if saved['architecture'][key] != model.architecture()[key]:
            raise ValueError('Stage transition architecture changed')
    if model.stage == 3 and saved['architecture']['qwen_targets'] != model.qwen_targets:
        raise ValueError('Previous-stage Qwen topology changed')
    for key in ('cache_sha256', 'contract'):
        if saved['identity'][key] != identity[key]:
            raise ValueError('Stage transition data/contract changed')
    current = model.state_dict()
    wanted = {k for k in current if k.startswith('extractor.') or (model.stage == 3 and k.startswith('qwen.'))}
    if wanted != set(saved['model']):
        raise ValueError('Previous stage weights missing or unexpected')
    if any(current[k].shape != v.shape or not torch.isfinite(v).all() for k, v in saved['model'].items()):
        raise ValueError('Previous stage tensor mismatch')
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    expected_missing = set(current) - wanted
    if set(missing) != expected_missing or unexpected:
        raise ValueError('Unexpected stage initialization gaps')
    return saved


def export_deployment(source, target):
    saved, _ = read(source)
    if saved['stage'] < 2:
        raise ValueError('Stage1 is not an AV generation checkpoint')
    state = {k: v for k, v in saved['model'].items() if not k.startswith(('extractor.video_decoder.', 'extractor.audio_decoder.'))}
    if not any(k.startswith('qwen.') for k in state):
        raise ValueError('Residual attention is missing')
    target = Path(target)
    atomic_save(target, {**saved, 'model': state, 'deployment_only': True})
    atomic_json_new(target.with_suffix('.complete.json'), dict(step=saved['step'], stage=saved['stage'], sha256=sha(target), world=0, optimizer_shards=[]))
