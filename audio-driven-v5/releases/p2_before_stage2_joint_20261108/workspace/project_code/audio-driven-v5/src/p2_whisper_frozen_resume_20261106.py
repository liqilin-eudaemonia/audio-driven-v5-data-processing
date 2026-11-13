"""Execute the checkpoint-pinned config from its verified archive, preserving workspace edits."""
import os
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha,atomic_json


def install():
    from p2_whisper_ablation import config as C
    if getattr(C,'_frozen_compat',False):return
    key=ROOT/'src/p2_whisper_event_feature/config.py'
    archive=ROOT/'experiments/p2_j_whisper_event_feature_20261105/optimized_snapshot/src/p2_whisper_event_feature/config.py'
    expected=read(ROOT/'experiments/p2_whisper_continue30_20261106/event/run.json')['source']
    if sha(archive)!=expected[str(key)]:raise ValueError('Archived config signature differs')
    current=key.read_text();pinned=archive.read_text()
    allowed=pinned.replace("if c['training']['save_every']!=500:\n        raise ValueError('This experiment saves complete checkpoints every500 updates')", "if type(c['training']['save_every']) is not int or c['training']['save_every']<1:\n        raise ValueError('Checkpoint interval must be a positive integer')")
    if current not in (pinned,allowed):raise ValueError('Unreviewed workspace config change')
    import p2_whisper_event_feature.config as module
    exec(compile(pinned,str(key),'exec'),module.__dict__)
    from p2_whisper_continue30 import install as native_install
    native_install()
    from p2_whisper_ablation import training as T
    def source():
        for name,h in expected.items():
            actual=archive if name==str(key) else Path(name)
            if sha(actual)!=h:raise ValueError('Frozen runtime source changed: '+name)
        return dict(expected)
    C.source_identity=T.source_identity=source;C._frozen_compat=True
    source()
    if os.environ.get('RANK') in (None,'0'):
        atomic_json(ROOT/'experiments/p2_whisper_continue30_20261106/FROZEN_SOURCE_COMPATIBILITY.json',dict(
            workspace=str(key),workspace_sha256=sha(key),executed_archive=str(archive),executed_sha256=sha(archive),
            reason='Unrelated checkpoint interval validation relaxation; run archived original code for exact checkpoint identity',
            workspace_preserved=True,bootstrap_sha256=sha(__file__)))
