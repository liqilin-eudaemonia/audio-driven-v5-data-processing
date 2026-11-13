import copy
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

import pytest
from p2_listen.contracts import ROOT,atomic_json,sha

spec=importlib.util.spec_from_file_location('single_aux_queue',ROOT/'scripts/queue_p2_whisper_ablation31.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


@pytest.mark.parametrize('arm',['event','feature'])
def test_monitor_accepts_only_enabled_loss_and_four_groups(arm):
    row=dict(av=2.,event=3. if arm=='event' else 0.,feature_video=4. if arm=='feature' else 0.,
             feature_audio=5. if arm=='feature' else 0.,seconds=6.,
             grad_norm={k:1. for k in ['connector','text_lora','whisper','main',arm+'s']})
    row['weighted_total']=row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
    queue.verify_row(row,arm)
    bad=copy.deepcopy(row);bad['event' if arm=='feature' else 'feature_audio']=1.
    bad['weighted_total']=bad['av']+.05*bad['event']+.025*(bad['feature_video']+bad['feature_audio'])
    with pytest.raises(ValueError,match='Disabled'):queue.verify_row(bad,arm)
    bad=copy.deepcopy(row);bad['grad_norm']['whisper']=0
    with pytest.raises(ValueError,match='gradient'):queue.verify_row(bad,arm)


def test_resume_requires_complete_same_file_and_ignores_uncommitted_newer_file(tmp_path):
    assert queue.resume_point(tmp_path) is None
    atomic_json(tmp_path/'run.json',{})
    with pytest.raises(RuntimeError,match='no complete'):queue.resume_point(tmp_path)
    committed=tmp_path/'step_000500.pt';committed.write_bytes(b'complete')
    atomic_json(committed.with_suffix('.complete.json'),dict(step=500,sha256=sha(committed)))
    (tmp_path/'step_001000.pt').write_bytes(b'incomplete write')
    assert queue.resume_point(tmp_path)==committed
    committed.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):queue.resume_point(tmp_path)


def test_tail_reader_ignores_partial_write(tmp_path):
    path=tmp_path/'metrics.jsonl'
    assert queue.recent_rows(path)==[]
    path.write_text('{"step": 1}\n{"step":')
    assert queue.recent_rows(path)==[dict(step=1)]


@pytest.mark.parametrize('stop_after_first',[False,True])
def test_queue_transitions_only_after_current_arm_finishes(monkeypatch,tmp_path,stop_after_first):
    from p2_whisper_ablation import config,data
    import listen_cluster
    import p2_seamless200_whisper_eval
    import p2_whisper_av.bootstrap
    reference=ROOT/'experiments/p2_whisper_event_20261106/cluster.json'
    from p2_listen.contracts import read
    cluster=read(reference);work=tmp_path/'queue';configs={}
    for arm in ('event','feature'):
        c=read(ROOT/f'configs/p2_whisper_{arm}_20261106.json');out=tmp_path/arm;out.mkdir()
        c['output']=str(out);configs[arm]=c
        atomic_json(out/'cluster.json',cluster)
        atomic_json(out/'preflight.json',dict(source={'pinned':'h'},data_identity={'cache':'h'}))
    monkeypatch.setattr(p2_whisper_av.bootstrap,'activate_ltx',lambda:None)
    monkeypatch.setattr(config,'load',lambda p:configs['feature' if 'feature' in str(p) else 'event'])
    monkeypatch.setattr(config,'validate_baseline',lambda *a:None)
    monkeypatch.setattr(config,'source_identity',lambda:{'pinned':'h'})
    monkeypatch.setattr(data,'data_identity',lambda c:{'cache':'h'})
    monkeypatch.setattr(p2_seamless200_whisper_eval,'idle_snapshot',lambda c:{'all_idle':True})
    monkeypatch.setattr(queue,'final_check',lambda out,c:(out/'status.json').exists())
    observed=[]
    def launch(c,commands,logs,cancel_event):
        assert sum(len(n['gpus']) for n in c['nodes'])==31
        assert c['nodes'][-1]['gpus']==list(range(7))
        arm='feature' if '/feature/' in str(logs) else 'event';observed.append(arm)
        atomic_json(Path(configs[arm]['output'])/'status.json',dict(step=8000,complete=True))
        if stop_after_first:(work/'STOP').touch()
    monkeypatch.setattr(listen_cluster,'launch',launch)
    queue.run(work)
    assert observed==(['event'] if stop_after_first else ['event','feature'])
    assert read(work/'state.json')['stage']==('stopped' if stop_after_first else 'complete')
