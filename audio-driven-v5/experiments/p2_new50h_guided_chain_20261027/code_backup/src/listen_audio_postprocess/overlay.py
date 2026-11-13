"""Compose immutable base video caches with separately stored audio updates."""
import copy
from pathlib import Path
from p2_listen.contracts import read,sha,digest,path_at,safe_id
from p2_listen.cache import load_part,save_part,tree_to
from p2_listen import features
from p2_listen.config import feature_loss_enabled
from .recipes import read_listener_audio,implementation_identity

VERSION='listener-audio-overlay-v1'


def settings(c):
    value=c['data'].get('audio_postprocess')
    if not isinstance(value,dict) or set(value)!={'schema','base_output','policy_manifest_sha256'} or value['schema']!=VERSION:
        raise ValueError('Invalid audio postprocess configuration')
    if feature_loss_enabled(c):raise ValueError('Audio overlay currently supports feature_loss_enabled=false only; stale teacher is forbidden')
    root=path_at(c['output']);base=path_at(value['base_output'])
    if root==base:raise ValueError('Audio overlay output must differ from base cache')
    manifest=root/'audio_postprocess/policy_manifest.json'
    if sha(manifest)!=value['policy_manifest_sha256']:raise ValueError('Postprocess policy manifest changed')
    return root,base


def sample_contract(c,obs,target,*,require_admitted=True):
    root,base=settings(c);sid=safe_id(obs['sample_id'])
    recipe=read(root/'audio_postprocess/recipes'/f'{sid}.json')
    if recipe['sample_id']!=sid or Path(recipe['source']['path']).resolve()!=Path(target['target_audio']).resolve() or recipe['source']['sha256']!=target['media_hashes']['target_audio']:
        raise ValueError('Recipe/source target mismatch')
    if abs(recipe['source']['samples']/recipe['source']['sample_rate']-obs['requested_duration_s'])>1/recipe['source']['sample_rate']+1e-6:
        raise ValueError('Recipe duration differs from observation')
    label=read(root/'labels/canonical'/f'{sid}.json')
    from p2_listen.annotations import validate,validate_contract
    validate(label,obs,require_admitted=require_admitted);validate_contract(label,c)
    if label['provenance'].get('audio_recipe_sha256')!=digest(recipe):raise ValueError('Label/recipe mismatch')
    folder=base/'cache/vae'/sid;marker=read(folder/'complete.json')
    expected=features.identity(c,'vae',obs,target)
    if marker['identity']!=digest(expected) or digest(marker['contract'])!=marker['identity']:
        raise ValueError('Base VAE contract differs')
    import p2_listen.runtime as runtime
    from . import encoder
    identity=dict(part='audio_postprocess',version=VERSION,recipe=recipe,
                  base_vae_identity=marker['identity'],base_payload_sha256=marker['payload_sha256'],
                  code={**implementation_identity(),str(Path(__file__)):sha(__file__),str(Path(encoder.__file__)):sha(encoder.__file__),str(Path(runtime.__file__)):sha(runtime.__file__)})
    return root,folder,expected,recipe,label,identity


def compose(base,patch):
    if patch.get('reuse_base_audio'):return base
    if patch['geometry']!=base['geometry']:raise ValueError('Overlay geometry differs')
    if patch['audio_latents']['latents'].shape!=base['av']['audio_latents']['latents'].shape or patch['audio_support'].shape!=base['support']['audio'].shape:
        raise ValueError('Overlay audio geometry differs')
    # Share the existing video tensor; no copying or serializing video payload.
    return dict(base,av=dict(base['av'],audio_latents=patch['audio_latents']),
                support=dict(base['support'],audio=patch['audio_support']),
                audio_support_margin_s=patch['audio_support_margin_s'],audio_endpoint_tokens_removed=patch['audio_endpoint_tokens_removed'])


def build_one(c,obs,target,encoder_factory,*,require_admitted=True):
    root,base_folder,expected,recipe,label,identity=sample_contract(c,obs,target,require_admitted=require_admitted)
    destination=root/'cache/audio_postprocess'/obs['sample_id']
    # Each sample has a single writer across nodes; use the existing flock helper.
    from listen_benchmark.full_fusion import lock
    with lock(root/'audio_postprocess/locks'/f'{obs["sample_id"]}.lock') as acquired:
        if not acquired:return dict(sample_id=obs['sample_id'],busy=True)
        previous=load_part(destination,identity,missing_ok=True)
        if previous is not None:return dict(sample_id=obs['sample_id'],reused=True)
        base=load_part(base_folder,expected)
        wave,rate,meta=read_listener_audio(recipe)
        if meta['waveform_changed']:
            patch=encoder_factory().encode(wave,rate,base['geometry'],base['av']['latents']['latents'].shape)
            patch['reuse_base_audio']=False
        else:patch=dict(reuse_base_audio=True)
        patch['waveform']=meta
        compose(base,patch)
        save_part(destination,identity,patch)
        return dict(sample_id=obs['sample_id'],reused=False,audio_encoded=meta['waveform_changed'],saved_bytes=(destination/'payload.pt').stat().st_size)


def load_training_sample(c,obs,target,device='cpu'):
    root,base_folder,expected,recipe,label,identity=sample_contract(c,obs,target)
    x,layout=features.load_input(c,obs,device)
    base=load_part(base_folder,expected,device)
    patch=load_part(root/'cache/audio_postprocess'/obs['sample_id'],identity,device)
    events=load_part(features.folder(c,'events',obs['sample_id']),features.identity(c,'events',obs,label=label),device)
    return x,layout,None,compose(base,patch),events
