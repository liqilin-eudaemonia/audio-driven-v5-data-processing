"""Full optimizer/RNG resume is distinct from stage transition/warm initialization."""
import json
from pathlib import Path
import torch
from omni_av_p2.checkpoint import atomic_save, atomic_json_new
from omni_av_p3.distributed import sha
from omni_av_p21.checkpoint import validate_state
from . import VERSION
from .quality import QUALITY_VERSION


def checked(path):
    path = Path(path)
    marker = json.loads(path.with_suffix('.complete.json').read_text())
    if sha(path) != marker['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['step'] != marker['step']:
        raise ValueError('Checkpoint step mismatch')
    return data


def save(path, model, optimizer, step, identity, rank_rng):
    model.assert_ownership()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    validate_state(model, state)
    atomic_save(path, dict(version=VERSION, stage=model.stage, step=step, identity=identity,
                           model=state, optimizer=optimizer.state_dict(), rank_rng=rank_rng))
    atomic_json_new(Path(path).with_suffix('.complete.json'), dict(step=step, sha256=sha(path)))


def resume(path, model, optimizer, identity, transition=None):
    data = checked(path)
    identity_matches=data['identity']==identity
    if not identity_matches and transition is not None:
        # Explicit, checkpoint-specific deployment certificate. No automatic
        # relaxation: model/recipe/world/data must remain byte-for-byte identical.
        certificate=json.loads(Path(transition).read_text())
        expected=dict(data['identity'],code=certificate['target_code'],
                      input_integrity_sha256=sha(certificate['input_integrity_seal']))
        identity_matches=(certificate.get('version')=='p3.audit-fix-transition.v1'
            and certificate['from_checkpoint_sha256']==sha(path)
            and certificate['from_identity']==data['identity'] and expected==identity)
    if (data.get('version') != VERSION or data['stage'] != model.stage or not identity_matches
            or len(data['rank_rng']) != identity['world_size']):
        raise ValueError('Resume stage/world/data/recipe mismatch')
    validate_state(model, data['model'])
    model.load_state_dict(data['model'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    return data


def authorized_transition(authorization, path, data, manifest_sha):
    """An explicit user decision is distinct from a measured quality pass."""
    return (authorization.get('version') == 'p3.user-stage-transition.v1'
        and authorization.get('user_authorized') is True
        and authorization.get('from_stage') == 'C0' and authorization.get('to_stage') == 'A'
        and authorization.get('from_step') == data['step']
        and authorization.get('a_steps') == 1000
        and authorization.get('evaluation_policy') == 'parallel_nonblocking'
        and authorization.get('checkpoint') == str(Path(path).resolve())
        and authorization.get('checkpoint_sha256') == sha(path)
        and authorization.get('manifest_sha256') == manifest_sha
        and bool(authorization.get('user_instruction')))


def initialize(path, model, manifest_sha, quality_gate=None, engineering_gate=False, transition_authorization=None):
    data = checked(path)
    if model.stage == 'C0':
        # Existing pilot Connector100 is a warm start, never an optimizer resume.
        if data.get('identity', {}).get('trainable') != 'generation_connector_only':
            raise ValueError('C0 requires the selected pilot Connector checkpoint')
        state = data['connector']
    else:
        if data.get('version') != VERSION or data.get('stage') != 'C0':
            raise ValueError('A requires a C0 checkpoint')
        if data['identity']['manifest_sha256'] != manifest_sha:
            raise ValueError('Stage transition data contract mismatch')
        gate = json.loads(Path(quality_gate).read_text()) if quality_gate else {}
        if not engineering_gate and data['identity'].get('engineering_gate'):
            raise ValueError('Engineering checkpoints cannot initialize production A')
        explicit = (json.loads(Path(transition_authorization).read_text()) if transition_authorization else {})
        approved = authorized_transition(explicit, path, data, manifest_sha) if explicit else False
        if transition_authorization and not approved:
            raise ValueError('User transition authorization does not match checkpoint/data/stage')
        if not engineering_gate and not approved and (gate.get('quality_version') != QUALITY_VERSION or gate.get('passed') is not True or gate.get('checkpoint_sha256') != sha(path)
                or gate.get('manifest_sha256') != manifest_sha):
            raise ValueError('A needs a passing evaluation tied to this exact C0 checkpoint')
        state = {k.removeprefix('connector.'): v for k, v in data['model'].items()
                 if k.startswith('connector.')}
    validate_state(model.connector, state)
    model.connector.load_state_dict(state, strict=True)
    return data['step']
