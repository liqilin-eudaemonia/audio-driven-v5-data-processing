import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import p2_j_av_event_then_av as queue
from p2_listen.contracts import atomic_json,sha,read


def setup_queue(tmp_path,monkeypatch,complete=True):
    work=tmp_path/'queue';work.mkdir()
    event=tmp_path/'event';av=tmp_path/'av'
    event.mkdir();av.mkdir()
    monkeypatch.setattr(queue,'WORK',work);monkeypatch.setattr(queue,'NEW',event)
    cluster=read(ROOT/'configs/p2_j_av_only_32gpu_20261101.json')
    atomic_json(work/'cluster.json',cluster)
    atomic_json(work/'state.json',{'stage':'prepared'})
    job=dict(av_event_output=str(event),av_output=str(av),budget=8927)
    for name in ['av_event_config','av_config','av_event_checkpoint']:
        path=tmp_path/name;path.write_text('{}')
        job[name]=str(path);job[name+'_sha256']=sha(path)
    atomic_json(work/'job.json',job)
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:'143000\n'*8)
    monkeypatch.setattr(queue.subprocess,'run',Mock())
    from p2_matrix16 import checkpoint
    check=Mock(return_value={'step':8927});monkeypatch.setattr(checkpoint,'checked',check)
    calls=[]
    def launch(cluster,commands,logs):
        calls.append(commands)
        out=event if len(calls)==1 else av
        atomic_json(out/'status.json',dict(step=8927 if complete else 4202,complete=complete))
    monkeypatch.setattr(queue,'launch',launch)
    return work,event,av,calls,check


def test_completes_event_then_starts_fresh_av(tmp_path,monkeypatch):
    work,event,av,calls,check=setup_queue(tmp_path,monkeypatch)
    queue.run()
    assert read(work/'state.json')['stage']=='complete'
    assert len(calls)==2 and all(len(c)==4 for c in calls)
    assert all('--resume' in c and str(ROOT/'scripts/p2_j_duration_train.py') in c for c in calls[0])
    assert all('--resume' not in c and str(ROOT/'scripts/p2_j_av_only.py') in c for c in calls[1])
    assert check.call_count==2


def test_incomplete_event_does_not_start_av(tmp_path,monkeypatch):
    work,event,av,calls,check=setup_queue(tmp_path,monkeypatch,complete=False)
    queue.run()
    assert len(calls)==1 and check.call_count==0
    assert read(work/'state.json')['stage']=='av_event_stopped'


def test_invalid_final_checkpoint_blocks_next_stage(tmp_path,monkeypatch):
    work,event,av,calls,check=setup_queue(tmp_path,monkeypatch)
    check.side_effect=ValueError('Checkpoint checksum mismatch')
    with pytest.raises(ValueError,match='checksum'):queue.run()
    assert len(calls)==1 and read(work/'state.json')['stage']=='failed'


def test_queue_stop_prevents_launch(tmp_path,monkeypatch):
    work,event,av,calls,check=setup_queue(tmp_path,monkeypatch)
    (work/'STOP').touch();queue.run()
    assert not calls and read(work/'state.json')['stage']=='stopped_before_av_event'
