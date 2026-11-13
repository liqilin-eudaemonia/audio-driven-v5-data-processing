import copy
import hashlib
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import p2_j_training_watchdog as w


CFG=dict(startup_grace_s=240,stall_s=180)


def snapshot(phase='av_event',step=5000):
    nodes=[]
    for n in range(4):
        nodes.append(dict(ok=True,node=f'node{n}',gpus=[dict(free_mib=10000) for _ in range(8)],
            processes=[dict(role=phase,rank=n*8+i,local_rank=i) for i in range(8)]))
    return dict(both_complete=False,phase=phase,nodes=nodes,queue={},
                phases={phase:dict(step=step,metrics_mtime=999)})


def test_healthy_requires_all_32_unique_ranks_and_progress():
    s=snapshot();action,state=w.decide(s,{},CFG,1000)
    assert action=='healthy'
    s['nodes'][3]['processes'].pop()
    action,state=w.decide(s,state,CFG,1060);assert action=='confirming_missing_workers'
    action,state=w.decide(s,state,CFG,1120);assert action=='recover_missing_workers'


def test_real_stall_is_confirmed_even_with_full_gpu_workers():
    s=snapshot();_,state=w.decide(s,{},CFG,1000)
    action,state=w.decide(s,state,CFG,1181);assert action=='confirming_stall'
    action,state=w.decide(s,state,CFG,1241);assert action=='recover_stall'


def test_ssh_timeout_never_restarts_training():
    s=snapshot();s['nodes'][0]=dict(ok=False,node='unreachable')
    previous=dict(unhealthy_count=10)
    for now in [1000,2000,3000]:
        action,state=w.decide(s,previous,CFG,now)
        assert action=='observation_unavailable' and state==previous


def test_slow_startup_has_grace_period():
    s=snapshot()
    for n in s['nodes']:n['processes']=[]
    s['nodes'][0]['processes']=[dict(role='controller',age_s=100)]
    action,state=w.decide(s,dict(unhealthy_count=8),CFG,1000)
    assert action=='starting' and state['unhealthy_count']==0


def test_latest_valid_checkpoint_skips_partial_and_corrupt_files(tmp_path):
    def save(step,data,expected=None):
        p=tmp_path/f'step_{step:06d}.pt';p.write_bytes(data)
        w.write(p.with_suffix('.complete.json'),dict(step=step,sha256=expected or hashlib.sha256(data).hexdigest()))
    save(200,b'complete');save(400,b'corrupt','not-the-hash')
    (tmp_path/'step_000500.pt').write_bytes(b'partial')
    result=w.latest_checkpoint(tmp_path)
    assert result['step']==200 and len(result['rejected_newer'])==1


def test_checkpoint_completion_switches_stage_only_after_valid_final(tmp_path,monkeypatch):
    event=tmp_path/'event';av=tmp_path/'av';event.mkdir();av.mkdir()
    job=dict(av_event_output=str(event),av_output=str(av),budget=8927)
    monkeypatch.setattr(w,'QUEUE',tmp_path/'queue')
    monkeypatch.setattr(w,'inspect_cluster',lambda *a:[])
    w.write(event/'status.json',dict(step=8927,complete=True))
    assert w.observe(job,{},tmp_path,{})['phase']=='av_event'
    p=event/'step_008927.pt';p.write_bytes(b'complete-final')
    w.write(p.with_suffix('.complete.json'),dict(step=8927,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert w.observe(job,{},tmp_path,{})['phase']=='av_only'


def test_pid_reuse_and_unrelated_process_are_never_killed(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir()
    w.write(tmp_path/'config.json',dict(queue=str(queue)))
    job=dict(av_event_config='/event.json',av_config='/av.json');w.write(queue/'job.json',job)
    own=dict(pid=1,start_ticks=10,command_sha256='old',args=[str(ROOT/'scripts/p2_j_av_event_then_av.py'),'run'])
    monkeypatch.setattr(w,'process_info',lambda pid:dict(own,start_ticks=11))
    kill=Mock();monkeypatch.setattr(w.os,'kill',kill)
    assert w.terminate_owned(tmp_path,[own])==[];kill.assert_not_called()
    monkeypatch.setattr(w,'process_info',lambda pid:dict(own,args=['python','some_other_training.py']))
    assert w.terminate_owned(tmp_path,[own])==[];kill.assert_not_called()


def test_worker_process_match_excludes_guard_and_torchrun():
    job=dict(av_event_config='/event.json',av_config='/av.json')
    args=['python',str(ROOT/'scripts/p2_j_duration_train.py'),'train','--config','/event.json']
    assert w.classify_process(dict(args=args),job)=='av_event'
    assert w.classify_process(dict(args=['python','--guard',*args]),job) is None
    assert w.classify_process(dict(args=['python','-m','torch.distributed.run',*args]),job) is None


def test_handoff_waits_for_initialization_without_duplicate_launch():
    s=snapshot('av_only',0)
    for n in s['nodes']:n['processes']=[]
    s['nodes'][0]['processes']=[dict(role='controller',age_s=10000)]
    s['queue']=dict(stage='waiting_release_for_av_only',at=995)
    assert w.decide(s,{},CFG,1000)[0]=='starting'


@pytest.mark.parametrize('event_complete',[False,True])
def test_supervisor_resumes_latest_checkpoint_and_preserves_stage_order(tmp_path,monkeypatch,event_complete):
    queue=tmp_path/'queue';queue.mkdir();monkeypatch.setattr(w,'QUEUE',queue)
    event=tmp_path/'event';av=tmp_path/'av';event.mkdir();av.mkdir()
    w.write(tmp_path/'config.json',dict(queue=str(queue)))
    cluster=w.read(ROOT/'configs/p2_j_av_only_32gpu_20261101.json');w.write(queue/'cluster.json',cluster)
    job=dict(av_event_output=str(event),av_output=str(av),budget=8927)
    for key in ['av_event_config','av_config']:
        p=tmp_path/(key+'.json');w.write(p,dict(id=key))
        job[key]=str(p);job[key+'_sha256']=w.digest_file(p)
    w.write(queue/'job.json',job)
    def save(out,step):
        p=out/f'step_{step:06d}.pt';p.write_bytes(str(step).encode())
        w.write(p.with_suffix('.complete.json'),dict(step=step,sha256=w.digest_file(p)))
    save(event,8927 if event_complete else 4400)
    if event_complete:save(av,200)
    sys.path.insert(0,str(ROOT/'src'))
    from p2_matrix16 import checkpoint
    def checked(path):
        path=Path(path);key='av_config' if path.parent==av else 'av_event_config'
        return dict(step=int(path.stem.split('_')[1]),identity=dict(experiment=w.read(job[key])))
    monkeypatch.setattr(checkpoint,'checked',checked)
    monkeypatch.setattr(w.subprocess,'run',Mock())
    monkeypatch.setattr(w,'inspect_cluster',lambda *a:[dict(ok=True,gpus=[dict(free_mib=143000)]*8)]*4)
    calls=[]
    def launch(c,commands,logs):
        calls.append(commands)
        out=av if str(ROOT/'scripts/p2_j_av_only.py') in commands[0] else event
        save(out,8927)
    monkeypatch.setattr(w,'launch',launch)
    w.supervise(tmp_path)
    assert w.read(queue/'state.json')['stage']=='complete'
    if event_complete:
        assert len(calls)==1 and str(av/'step_000200.pt') in calls[0][0]
    else:
        assert len(calls)==2 and str(event/'step_004400.pt') in calls[0][0]
        assert '--resume' not in calls[1][0]


def test_four_hour_and_hourly_records_are_written(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir();monkeypatch.setattr(w,'QUEUE',queue)
    cfg=dict(CFG,started_at=0,minimum_until=36000,handoff_at=14400,poll_s=60)
    w.write(tmp_path/'config.json',cfg)
    w.write(queue/'job.json',dict(av_event_output=str(tmp_path/'event'),av_output=str(tmp_path/'av')))
    w.write(queue/'cluster.json',{})
    s=snapshot('av_only',200);s['phases']['av_only']['metrics_mtime']=14999
    s['phases']['av_only']['complete']=False
    s['phases']['av_event']=dict(step=8927,complete=True)
    monkeypatch.setattr(w,'observe',lambda *a:s)
    monkeypatch.setattr(w.time,'time',lambda:15000)
    w.tick(tmp_path)
    assert w.read(tmp_path/'FOUR_HOUR_CHECK.json')['status']=='av_only_running'
    assert w.read(tmp_path/'HANDOFF_VERIFIED.json')['status']=='av_only_running'
    assert w.read(tmp_path/'hourly/hour_04.json')['action']=='healthy'
