"""Read admitted postprocessed targets, preserving the two published producer identities."""
import copy
from pathlib import Path
from . import features
from .contracts import read,sha,digest
from .cache import load_part
from listen_audio_postprocess import overlay

# The only producer change added require_admitted=True plus a cache-only opt-out.
# Admitted-sample validation and audio construction are byte-for-byte unchanged.
CURRENT_OVERLAY_SHA='2a152ab6eb107686077744644c52aa14310d941a5daabec433dc97db953e3502'
TRAIN50H_OVERLAY_SHA='6e970966aca137d47ea22673836382a4e2bac459e8f37d0207d0f59abef76159'
LEGACY_SOURCE=Path(__file__).resolve().parents[2]/'compat/overlay_train50h.py'


def audio_cache_identity(expected,marker):
    if marker.get('identity')==digest(expected):return expected
    original=str(Path(overlay.__file__))
    # Never admit an arbitrary changed implementation, recipe, geometry, source,
    # base VAE, encoder or policy. Only this reviewed historical source is allowed.
    if (expected['code'].get(original)!=CURRENT_OVERLAY_SHA or sha(overlay.__file__)!=CURRENT_OVERLAY_SHA
            or sha(LEGACY_SOURCE)!=TRAIN50H_OVERLAY_SHA):
        raise ValueError('Unrecognized postprocessed audio producer version')
    compatible=copy.deepcopy(expected);compatible['code'][original]=TRAIN50H_OVERLAY_SHA
    if marker.get('identity')!=digest(compatible) or marker.get('contract')!=compatible:
        raise ValueError('Audio cache identity differs from current and verified train50h producers')
    return compatible


def load_training_sample(c,obs,target,device='cpu'):
    root,base_folder,base_identity,recipe,label,expected=overlay.sample_contract(c,obs,target)
    cache=root/'cache/audio_postprocess'/obs['sample_id']
    identity=audio_cache_identity(expected,read(cache/'complete.json'))
    x,layout=features.load_input(c,obs,device)
    base=load_part(base_folder,base_identity,device)
    patch=load_part(cache,identity,device)
    # Even reuse_base_audio=True must read the NEW label/event cache.
    events=load_part(features.folder(c,'events',obs['sample_id']),features.identity(c,'events',obs,label=label),device)
    return x,layout,None,overlay.compose(base,patch),events
