import copy
import json
from pathlib import Path
import shlex

import pytest

from p2_plan_cluster import cluster as cl


def test_exact_three_host_mapping_and_ports():
    c=cl.load();rows=cl.assignments(c)
    assert len(rows)==24 and [r['rank'] for r in rows]==list(range(24))
    assert [(rows[i]['node'],rows[i]['host'],rows[i]['gpu']) for i in (0,8,16)]==[
        ('node17','172.29.15.91',0),('node12','172.29.252.11',0),('node4','172.29.225.51',0)]
    for node in c['nodes']:
        cmd=cl.ssh(c,node,['status','--job','/tmp/path with spaces'])
        assert cmd[cmd.index('-p')+1]=='22'
        assert str(node['external_port_note']) not in cmd
        assert shlex.split(cmd[-1])[-1]=='/tmp/path with spaces'


def test_worker_commands_have_unique_shards_and_local_cuda_zero():
    c=cl.load()
    for r in cl.assignments(c):
        for stage in cl.STAGES:
            cmd=cl.worker_command(c,'/tmp/job',r['rank'],stage)
            assert cmd[cmd.index('--shards')+1]=='24'
            assert cmd[cmd.index('--shard-index')+1]==str(r['rank'])
            assert ('--device' in cmd)==(stage!='verify')
            if stage!='verify':assert cmd[-2:]==['--device','cuda:0']
            assert 'torch.distributed.run' not in cmd


@pytest.mark.parametrize('mutation',['gpu_duplicate','host_duplicate','port','retries','stages'])
def test_invalid_cluster_rejected(tmp_path,mutation):
    c=cl.load()
    if mutation=='gpu_duplicate':c['nodes'][0]['gpus']=[0,0]
    elif mutation=='host_duplicate':c['nodes'][1]['host']=c['nodes'][0]['host']
    elif mutation=='port':c['nodes'][0]['ssh_port']=0
    elif mutation=='retries':c['retries']=99
    else:c['stages'].reverse()
    path=tmp_path/'c.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError):cl.load(path)


def test_idle_checks_processes_even_when_utilization_zero():
    c=cl.load();g=dict(compute_process_present=False,used_mib=0,utilization=0)
    assert cl.idle(g,c)
    assert not cl.idle(dict(g,compute_process_present=True),c)
    assert not cl.idle(dict(g,used_mib=5000),c)
    assert not cl.idle(dict(g,utilization=90),c)


def test_shared_admission_uses_free_memory_and_keeps_reserve():
    c=cl.load();c['gpu_policy']=dict(mode='shared',min_free_mib=65536,process_limit_gib=48)
    row=dict(compute_process_present=True,used_mib=20000,utilization=100,free_mib=90000)
    assert cl.available(row,c)
    assert not cl.available(dict(row,free_mib=60000),c)


def job(tmp_path,smoke=True):
    c=cl.load();p=tmp_path/'job';p.mkdir()
    cl.atomic_json(p/'job.json',dict(config=c,source=cl.source_identity(),smoke=smoke,
        assignments=cl.assignments(c),protocol_sha256=None))
    return p


def test_smoke_partition_coverage_and_tamper(tmp_path,monkeypatch):
    p=job(tmp_path)
    for a in cl.assignments(cl.load()):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES',str(a['gpu']))
        for stage in cl.STAGES:cl.smoke_worker(p,a['rank'],stage)
    assert cl.smoke_check(p)==dict(success=True,workers=24,stages=3,samples=257,gpu_used=False)
    path=p/'smoke/verify/000.json';v=cl.read(path);v['ids'].append('sample-0000');cl.atomic_json(path,v)
    with pytest.raises(ValueError,match='coverage'):cl.smoke_check(p)


def test_status_requires_every_node_and_worker_success(tmp_path):
    p=job(tmp_path)
    for a in cl.assignments(cl.load()):cl.atomic_json(p/'workers'/f'{a["rank"]:03d}.json',dict(**a,state='succeeded'))
    assert not cl.status(p)['success']
    for n in cl.load()['nodes']:cl.atomic_json(p/'nodes'/(n['node']+'.json'),dict(node=n['node'],state='succeeded'))
    assert cl.status(p)['success']
    cl.atomic_json(p/'workers/023.json',dict(state='failed'))
    assert cl.status(p)['state']=='failed'


def test_source_change_rejected(tmp_path,monkeypatch):
    p=job(tmp_path)
    monkeypatch.setattr(cl,'source_identity',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='source changed'):cl.check_job(p)


def test_busy_probe_cannot_launch_real_workers(tmp_path,monkeypatch):
    c=cl.load();p=tmp_path/'job'
    from p2_plan_cache import protocol
    monkeypatch.setattr(protocol,'open_protocol',lambda *args:dict(config=cl.read(c['data_config'])))
    monkeypatch.setattr(cl,'sha',lambda *args:'digest')
    monkeypatch.setattr(cl,'probe',lambda *args:dict(success=True,all_idle=False,output='/probe'))
    monkeypatch.setattr(cl,'remote',lambda *args:pytest.fail('Must not launch a worker'))
    with pytest.raises(RuntimeError,match='occupied'):cl.start(c,p)
    assert not (p/'job.json').exists()


@pytest.mark.parametrize('persistent_failure',[False,True])
def test_supervisor_stage_order_and_bounded_retry(tmp_path,monkeypatch,persistent_failure):
    p=job(tmp_path);calls=[]
    class Child:
        def __init__(self,cmd,**kwargs):
            rank=int(cmd[cmd.index('--rank')+1]);stage=cmd[cmd.index('--stage')+1]
            attempts=sum(1 for r,s in calls if (r,s)==(rank,stage))
            calls.append((rank,stage));self.pid=10000+len(calls)
            self.rc=7 if rank==0 and stage=='cache-text' and (persistent_failure or attempts==0) else 0
        def poll(self):return self.rc
    monkeypatch.setattr(cl.subprocess,'Popen',Child)
    monkeypatch.setattr(cl.time,'sleep',lambda *args:None)
    monkeypatch.setattr(cl.signal,'signal',lambda *args:None)
    result=cl.supervise(p,'node17')
    assert result['success'] is not persistent_failure
    assert calls.count((0,'cache-text'))==2
    if persistent_failure:
        assert (0,'cache-input-plan') not in calls
        assert cl.read(p/'workers/000.json')['state']=='failed'
    else:
        assert [s for r,s in calls if r==0]==['cache-text','cache-text','cache-input-plan','verify']
    for rank in range(1,8):
        assert [s for r,s in calls if r==rank]==list(cl.STAGES)
        assert cl.read(p/'workers'/f'{rank:03d}.json')['state']=='succeeded'


def test_detached_launch_reports_success_to_ssh_caller(tmp_path,monkeypatch):
    from types import SimpleNamespace
    p=job(tmp_path)
    monkeypatch.setattr(cl.subprocess,'Popen',lambda *args,**kwargs:SimpleNamespace(pid=123))
    result=cl.detach(p,'node17')
    assert result['success'] and result['state']=='launched'
    with cl.node_lock(p,'node17'):
        duplicate=cl.detach(p,'node17')
    assert duplicate['success'] and duplicate['state']=='already_running'
