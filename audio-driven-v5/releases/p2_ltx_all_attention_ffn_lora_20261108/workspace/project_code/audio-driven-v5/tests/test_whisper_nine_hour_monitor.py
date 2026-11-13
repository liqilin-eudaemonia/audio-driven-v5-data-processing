import copy
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ninehour',ROOT/'scripts/monitor_p2_whisper_nine_hours.py')
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def healthy():
    return dict(at=10000,last_metrics=dict(av=1.,av_video=.5,av_audio=.5,weighted_total=1.,event=0.,feature_video=0.,feature_audio=0.),
        invalid_metric_lines=0,training_complete=False,stop_requested=False,guard_alive=True,controller_alive=True,
        queue=dict(phase='training'),metrics_age_s=5,free_disk_bytes=1000*2**30,controller_started_at=0,
        recent200_count=200,recent200_mean_seconds=5.)


def test_live_jobs_and_normal_checkpoint_pause_never_request_restart():
    s=healthy();assert monitor.issues(s)==[]
    s['metrics_age_s']=240;assert monitor.issues(s,checkpoint_writing=True)==[]
    assert monitor.issues(s)==['no_recent_steps_diagnose_only']
    assert monitor.ensure_guard({},s,None) is None


def test_new_controller_has_bounded_startup_grace():
    s=healthy();s.update(controller_started_at=9950,metrics_age_s=300)
    assert monitor.issues(s)==[]
    s.update(at=11000,metrics_age_s=700)
    assert 'no_recent_steps_diagnose_only' in monitor.issues(s)
    s=healthy();s.update(controller_started_at=9800,metrics_age_s=180)
    assert 'no_recent_steps_diagnose_only' in monitor.issues(s)


def test_requested_stop_and_complete_dont_launch_guard():
    s=healthy();s.update(guard_alive=False,controller_alive=False,stop_requested=True)
    assert monitor.issues(s)==[] and monitor.ensure_guard({},s,None) is None
    s.update(stop_requested=False,training_complete=True)
    assert monitor.issues(s)==[] and monitor.ensure_guard({},s,None) is None


def test_partially_written_last_metric_is_not_a_failed_training_record(tmp_path):
    p=tmp_path/'metrics.jsonl';p.write_text(json.dumps({'step':1})+'\n{"step":')
    assert monitor.metrics(p)==([{'step':1}],0)
    p.write_text('bad\n'+json.dumps({'step':2})+'\n')
    assert monitor.metrics(p)==([{'step':2}],1)


def test_nonfinite_wrong_objective_and_slow_progress_are_reported():
    s=healthy();s['last_metrics']['av']=float('nan')
    assert 'loss_contract_violation' in monitor.issues(s)
    s=healthy();s['last_metrics']['event']=.1
    assert 'loss_contract_violation' in monitor.issues(s)
    s=healthy();s['recent200_mean_seconds']=8
    assert 'sustained_slow_steps_diagnose_only' in monitor.issues(s)


def test_remote_probe_uses_absolute_work_directory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path);work=tmp_path/'relative';work.mkdir();seen=[]
    def transport(cluster,node,command):
        seen.append(command[command.index('--work')+1]);return command
    monkeypatch.setattr(monitor,'transport',transport)
    monkeypatch.setattr(monitor.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='{"ok":true}',stderr=''))
    c=dict(python='python',nodes=[dict(name=f'node{i}') for i in range(4)])
    assert all(row['ok'] for row in monitor.inspect_cluster({},c,Path('relative')))
    assert seen==[str(work)]*4
