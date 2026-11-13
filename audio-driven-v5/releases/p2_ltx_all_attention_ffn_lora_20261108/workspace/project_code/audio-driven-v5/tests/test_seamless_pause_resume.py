import importlib.util
import json
from pathlib import Path

import pytest

P=Path(__file__).resolve().parents[1]/'scripts/p2_seamless200_pause_resume.py'
s=importlib.util.spec_from_file_location('seamless_pause',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


@pytest.mark.parametrize('manual_stop',[False,True])
def test_waits_for_another_job_then_stable_idle(tmp_path,monkeypatch,manual_stop):
    folder=tmp_path/'pause_for_other_job';folder.mkdir()
    (folder/'request.json').write_text(json.dumps(dict(token='ours',idle_seconds=2,poll_seconds=1)))
    (tmp_path/'STOP').write_text(json.dumps(dict(token='manual' if manual_stop else 'ours')))
    snapshots=iter([dict(idle=True,external_busy=False),dict(idle=False,external_busy=True),
                    dict(idle=True,external_busy=False),dict(idle=True,external_busy=False),dict(idle=True,external_busy=False)])
    now=[0];launched=[]
    monkeypatch.setattr(m,'observe',lambda:next(snapshots))
    monkeypatch.setattr(m.time,'time',lambda:now[0])
    monkeypatch.setattr(m.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+seconds))
    class Process:
        pid=123
    def launch(*a,**k):
        assert now[0]>=4
        launched.append(a);return Process()
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    m.wait(tmp_path)
    if manual_stop:
        assert not launched and (tmp_path/'STOP').exists()
        assert m.read(folder/'state.json')['stage']=='blocked_manual_stop_changed'
    else:
        assert len(launched)==2 and not (tmp_path/'STOP').exists()
        assert m.read(folder/'RESUMED.json')['external_job_observed']
