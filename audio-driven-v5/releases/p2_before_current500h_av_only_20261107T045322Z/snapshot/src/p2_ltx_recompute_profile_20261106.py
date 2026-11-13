"""Strict same-loss 32-rank recomputation profile; native files remain unchanged."""
import copy
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
PARENT=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'
BASE=ROOT/'experiments/p2_ltx_clip_recompute32_20261106'
PARENT_SHA='f509abbdabf680d6da2c04c0d30cd7edabb01e341d8f8c199753c77a1786e360'


def normalize_profile(c):
    path=PARENT/'config.json'
    if sha(path)!=PARENT_SHA:raise ValueError('Frozen recomputation parent changed')
    parent=read(path);expected=copy.deepcopy(parent)
    expected['output']=str(BASE)
    expected['training']['gradient_checkpointing']='three_quarters'
    expected['recompute_profile']=dict(parent_config=str(path),parent_sha256=PARENT_SHA,policy='three_quarters',world=32)
    if c!=expected:raise ValueError('Recomputation profile changed data/loss/optimizer/schedule or world')
    return parent


def install():
    from p2_ltx_events import config as C,training as T
    if getattr(C,'_recompute32_profile_installed',False):return
    old_validate=C.validate_baseline;old_source=C.source_identity
    def validate_baseline(c,parent):
        if 'recompute_profile' in c:return old_validate(normalize_profile(c),parent)
        return old_validate(c,parent)
    def source_identity():
        result=old_source()
        for p in (Path(__file__),ROOT/'scripts/p2_ltx_recompute_profile32_20261106.py'):result[str(p)]=sha(p)
        return result
    C.validate_baseline=T.validate_baseline=validate_baseline
    C.source_identity=T.source_identity=source_identity
    C._recompute32_profile_installed=True
