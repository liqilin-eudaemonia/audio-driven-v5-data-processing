"""Validated Event-only recomputation profile; preserve checkpoint-pinned native code."""
import copy
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
BASE=ROOT/'experiments/p2_whisper_recompute30_adopted_20261106'
PARENT=ROOT/'experiments/p2_whisper_continue30_20261106'


def normalize_profile(c):
    arm=c['loss'][3:]
    path=PARENT/arm/'config.json';parent=read(path)
    expected=copy.deepcopy(parent);expected['output']=str(BASE/arm)
    if arm=='event':expected['training']['gradient_checkpointing']='three_quarters'
    expected['recompute_profile']=dict(parent_config=str(path),parent_sha256=sha(path),policy=expected['training']['gradient_checkpointing'])
    if c!=expected:raise ValueError('Unapproved recompute profile change')
    return parent


def install():
    from p2_whisper_frozen_resume_20261106 import install as frozen
    frozen()
    from p2_whisper_ablation import config as C,training as T
    import p2_whisper_continue30 as native
    if getattr(C,'_recompute_profile_20261106',False):return
    old_normalized=native.normalized;old_source=C.source_identity
    def normalized(c):
        if 'recompute_profile' in c:return old_normalized(normalize_profile(c))
        return old_normalized(c)
    def source():
        v=old_source()
        for p in [Path(__file__),ROOT/'scripts/p2_whisper_recompute_profile30_20261106.py']:v[str(p)]=sha(p)
        return v
    native.normalized=normalized;C.source_identity=T.source_identity=source
    C._recompute_profile_20261106=True
