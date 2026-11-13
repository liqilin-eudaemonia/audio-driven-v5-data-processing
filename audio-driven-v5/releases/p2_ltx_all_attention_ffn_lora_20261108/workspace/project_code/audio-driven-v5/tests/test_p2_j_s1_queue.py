import copy
import sys

import pytest
from p2_listen.contracts import ROOT, read, atomic_json

sys.path.insert(0,str(ROOT/'scripts'))
import p2_j_s1_queue as queue
import listen_cluster


def cluster():
    return dict(python='/python',master_port=29761,network={'OMP_NUM_THREADS':'2'},
        nodes=[dict(name=f'n{i}',host=f'10.0.0.{i+1}',gpus=list(range(8))) for i in range(4)])


def test_four_node_commands_share_config_budget_and_resume():
    c=cluster();commands=queue.commands(c,'/config',stop_after=16)
    assert len(commands)==4
    for rank,command in enumerate(commands):
        assert '--nnodes=4' in command and '--nproc_per_node=8' in command
        assert '--node_rank='+str(rank) in command
        assert '--master_addr=10.0.0.1' in command
        assert command[-2:]==['--stop-after','16']
        assert command[command.index('--config')+1]=='/config'
    resumed=queue.commands(c,'/config',resume='/step16.pt')
    assert all(cmd[-2:]==['--resume','/step16.pt'] and '--stop-after' not in cmd for cmd in resumed)


def test_transport_uses_node_key_and_recorded_known_hosts(monkeypatch):
    monkeypatch.setattr(listen_cluster.subprocess,'check_output',lambda *a,**kw:'127.0.0.1')
    cfg=dict(ssh_key='/default',known_hosts='/known')
    node=dict(host='192.0.2.2',ssh_user='root',ssh_port=22,ssh_key='/node12')
    cmd=listen_cluster.transport(cfg,node,['python','script.py'])
    assert cmd[cmd.index('-i')+1]=='/node12'
    assert 'UserKnownHostsFile=/known' in cmd


def test_initial_training_requires_complete_steps_finite_losses_and_gradients():
    rows=[dict(step=i,seconds=5.,peak_reserved_gib=90.,av=1.,av_video=.5,av_audio=.5,event=.2,
        weighted_total=1.01,grad_norm=dict(connector=.1,events=.2,text_lora=.3)) for i in range(1,17)]
    assert queue.verify_metrics(rows,16)['steps']==16
    with pytest.raises(ValueError):queue.verify_metrics(rows[:-1],16)
    bad=copy.deepcopy(rows);bad[3]['av']=float('nan')
    with pytest.raises(ValueError):queue.verify_metrics(bad,16)
    bad=copy.deepcopy(rows)
    for r in bad:r['grad_norm']['connector']=0
    with pytest.raises(ValueError):queue.verify_metrics(bad,16)


@pytest.mark.parametrize('complete',[False,True])
def test_queue_waits_for_audit_and_gpu_guard_release(tmp_path,monkeypatch,complete):
    c=read(ROOT/'configs/p2_j_clip_events_20261101/guided_av_event_s1.json')
    cache=tmp_path/'cache';out=tmp_path/'run';c.update(j_cache=str(cache),output=str(out))
    config=tmp_path/'config.json';atomic_json(config,c)
    atomic_json(cache/'status.json',dict(phases=dict(features=dict(samples=12))))
    if complete:atomic_json(cache/'COMPLETE.json',dict(success=True))
    cc=cluster()
    for node in cc['nodes']:atomic_json(cache/'launchers'/(node['name']+'.json'),dict(status='running'))
    monkeypatch.setattr(queue,'launch',lambda *a,**kw:pytest.fail('Training started before cache released GPUs'))
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:(out/'STOP').touch())
    queue.run(config,cc)
    assert read(out/'queue/state.json')['state']=='stopped'
