"""Checkpoint-pinned LTX inference and a lightweight, audited latest-SFT session."""
import importlib
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha
TRAIN=ROOT/'experiments/p2_ltx_clip_train31_cadence_20261107'
ARCHIVE=ROOT/'experiments/p2_ltx_clip_train24_reserve_node11_20261107/source_before'


def install():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_ltx_events import config as C
    if getattr(C,'_eval6000_pinned',False):return
    expected=read(TRAIN/'run.json')['source'];executed={}
    # Execute the actual checkpoint-pinned code, preserving the live24-card
    # workspace. No new24-card schedule module is used in this inference path.
    for name in ['config','resume31','schedule']:
        path=ROOT/f'src/p2_ltx_events/{name}.py';archive=ARCHIVE/path.relative_to(ROOT)
        if sha(archive)!=expected[str(path)]:raise ValueError('Archive differs from checkpoint: '+name)
        module=importlib.import_module('p2_ltx_events.'+name)
        exec(compile(archive.read_bytes(),str(path),'exec'),module.__dict__)
        executed[str(path)]=dict(archive=str(archive),sha256=sha(archive),workspace_sha256=sha(path))
    def source():
        for name,h in expected.items():
            path=Path(executed[name]['archive']) if name in executed else Path(name)
            if sha(path)!=h:raise ValueError('Pinned inference source changed: '+name)
        return dict(expected)
    C.source_identity=source;C._eval6000_pinned=True;C._eval6000_executed=executed
    source()


class SFTInferenceSession:
    """Only inference assets, with the API consumed by native load_session.

    No training targets, teacher or obsolete2697-step hidden cache are opened.
    """
    def __init__(self,work):
        from types import SimpleNamespace
        from p2_j_sft_hidden import adapter_identity
        p=read(Path(work)/'protocol.json')
        if p['schema']!='latest-sft-inference-session-v1':raise ValueError('Wrong SFT session')
        if adapter_identity(p['adapter']['path'])!=p['adapter']:raise ValueError('SFT weights changed')
        if sha(p['training_complete'])!=p['training_complete_sha256'] or read(p['training_complete'])['global_step']!=2752:raise ValueError('Wrong latest SFT completion')
        self.protocol=p;self.source=SimpleNamespace(manifest=dict(config=p['data_config']))
