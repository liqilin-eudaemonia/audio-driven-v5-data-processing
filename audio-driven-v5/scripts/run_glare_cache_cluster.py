#!/usr/bin/env python3
"""Prepare/preflight/start/status/stop the explicitly allocated four-node28GPU cache job.

prepare, commands and preflight do not start GPU workloads. Only start launches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster
from listen_cluster import transport


def remote(cfg,node,args,timeout=600):
    ssh=cluster.read(cfg['ssh_config'])
    result=subprocess.run(transport(ssh,node,[cfg['python'],str(Path(__file__).resolve()),'--config',
                           str(CONFIG_PATH),*args]),text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(node['name']+': '+result.stderr[-5000:]+result.stdout[-2000:])
    return json.loads(result.stdout)


def commands(cfg):
    return [dict(node=n['name'],gpus=n['gpus'],command=shlex.join([cfg['python'],
        str(ROOT/'scripts/glare_cache_cluster_node.py'),'--config',str(CONFIG_PATH),'--node',n['name']])) for n in cfg['nodes']]


def check_nodes(cfg,full_assets=False):
    def check(node):
        try:return remote(cfg,node,['check-node','--node',node['name'],*(['--full-assets'] if full_assets else [])])
        except Exception as exc:return dict(node=node['name'],ready=False,errors=[repr(exc)])
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(check,cfg['nodes']))
    cluster.save(Path(cfg['work'])/'preflight.json',dict(nodes=results,checked_at=time.time(),gpu_work_started=False))
    return results


def launch_node(cfg,m,name):
    node=cluster.node_by_name(cfg,name);cluster.assert_local(node);cluster.assert_active(cfg,m)
    folder=Path(cfg['work'])/'nodes'/name
    # Existing supervisor prevents duplicate children on repeated start.
    try:
        with cluster.locked(folder/'supervisor.lock'):pass
    except BlockingIOError:return dict(node=name,already_running=True)
    if (Path(cfg['work'])/'STOP').exists():raise RuntimeError('STOP exists')
    busy=[g['index'] for g in cluster.gpu_info() if g['index'] in node['gpus'] and
          g['used_mib']>cfg['preflight']['max_used_gpu_memory_mib']]
    if cfg['preflight']['require_empty_assigned_gpus'] and busy:
        raise RuntimeError('Assigned cards busy on '+name+': '+str(busy)+'; external tasks preserved')
    cmd=[cfg['python'],str(ROOT/'scripts/glare_cache_cluster_node.py'),'--config',str(CONFIG_PATH),'--node',name]
    with (folder/'supervisor.log').open('a') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(node=name,pid=p.pid,command=cmd,launched_at=time.time(),run_id=m['run_id'])
    cluster.save(folder/'launch.json',result)
    # A successful SSH response must mean the detached supervisor actually holds
    # the cache fences, not just that fork() succeeded.
    for _ in range(100):
        if p.poll() is not None:raise RuntimeError('Node supervisor failed; inspect '+str(folder/'supervisor.log'))
        path=folder/'status.json'
        if path.exists():
            state=cluster.read(path)
            if state.get('pid')==p.pid and state['run_id']==m['run_id']:return result
        time.sleep(.1)
    raise RuntimeError('Node supervisor startup timed out; inspect its log before retry')


def start(cfg,m):
    work=Path(cfg['work']);old=Path(cfg['shared_lia'])
    with cluster.locked(old/'cluster_owner.lock'):
        marker=old/'cluster_active.json'
        # Exclusive probes block legacy writers and other runs. Same-run restart
        # can adopt still-running supervisors, whose shared fences remain held.
        same=marker.exists() and cluster.read(marker)['run_id']==m['run_id']
        try:
            guards=cluster.writer_fences(cfg,m,shared=False);handles=guards.__enter__()
        except BlockingIOError:
            if not same:raise RuntimeError('Other GLARE cache writers are active; no task launched')
            guards=cluster.writer_fences(cfg,m,shared=True);handles=guards.__enter__()
        try:
            checked=check_nodes(cfg,full_assets=True)
            assert all(not n.get('errors') for n in checked),'Preflight failed; no new workers launched'
            if not same and cfg['preflight']['require_empty_assigned_gpus']:
                assert all(n['ready'] for n in checked),'Assigned cards busy; no task launched'
            cluster.save(marker,dict(run_id=m['run_id'],work=str(work),at=time.time()))
            for f in handles:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_SH)
            # Clear only this run's stop request after every validation passed.
            (work/'STOP').unlink(missing_ok=True)
            launched=[remote(cfg,n,['launch-node','--node',n['name']]) for n in cfg['nodes']]
            cluster.save(work/'launch.json',dict(run_id=m['run_id'],nodes=launched,started=time.time()))
            return dict(launched=launched)
        finally:guards.__exit__(None,None,None)


def status(cfg,m):
    work=Path(cfg['work'])
    result=dict(run_id=m['run_id'],at=time.time(),nodes={},stopped=(work/'STOP').exists(),
                queue={s:cluster.TaskQueue(work,m['run_id'],s).summary() for s in cluster.STAGES})
    for n in cfg['nodes']:
        path=work/'nodes'/n['name']/'status.json'
        result['nodes'][n['name']]=cluster.read(path) if path.exists() else dict(stage='not_started')
        slots={}
        for gpu in n['gpus']:
            for worker_slot in range(cluster.execution(cfg)['lia_workers_per_gpu']):
                folder=cluster.slot_dir(cfg,n,gpu,worker_slot);active=folder/'active.json'
                if not active.exists():continue
                task=cluster.read(active);stage=task['stage']
                state_path=folder/('status.json' if stage=='prosody' else 'workers/gpu%d.json'%gpu)
                state=cluster.read(state_path) if state_path.exists() else {}
                slots['%d/%d'%(gpu,worker_slot)]=dict(task=task,state=state)
        result['nodes'][n['name']]['slots']=slots
    result['complete']=all(q['complete'] for q in result['queue'].values())
    cluster.save(work/'status.json',result);return result


def main():
    global CONFIG_PATH
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=cluster.DEFAULT)
    ap.add_argument('action',choices=['prepare','commands','preflight','start','stop','status','retry-errors','assemble','check-node','launch-node'])
    ap.add_argument('--node');ap.add_argument('--full-assets',action='store_true')
    a=ap.parse_args();CONFIG_PATH=a.config.resolve();cfg=cluster.config(CONFIG_PATH)
    if a.action=='prepare':
        m=cluster.prepare(cfg);result=dict(prepared=True,run_id=m['run_id'],pairs=m['pairs'],chunks=m['chunks'],gpu_work_started=False)
    elif a.action=='commands':result=dict(commands=commands(cfg),gpu_work_started=False)
    else:
        m=cluster.load_run(cfg)
        if a.action=='preflight':result=dict(nodes=check_nodes(cfg,a.full_assets),gpu_work_started=False)
        elif a.action=='check-node':result=cluster.preflight(cfg,m,cluster.node_by_name(cfg,a.node),a.full_assets)
        elif a.action=='launch-node':result=launch_node(cfg,m,a.node)
        elif a.action=='start':result=start(cfg,m)
        elif a.action=='stop':
            cluster.save(Path(cfg['work'])/'STOP',dict(requested_at=time.time(),reason='explicit cluster stop'))
            result=dict(stop_requested=True,scope='this run only; supervisors drain their own children')
        elif a.action=='status':result=status(cfg,m)
        elif a.action=='retry-errors':
            with cluster.locked(Path(cfg['shared_lia'])/'cluster_owner.lock'):
                result=dict(reset={s:cluster.TaskQueue(cfg['work'],m['run_id'],s).retry_errors() for s in cluster.STAGES})
        elif a.action=='assemble':
            assert not cfg.get('benchmark'),'Benchmark output cannot be published as production training caches'
            current=status(cfg,m)
            assert current['complete'] and all(n['stage']=='complete' for n in current['nodes'].values()),'Queues and supervisors must be complete before assembly'
            subprocess.run([cfg['python'],str(ROOT/'scripts/prepare_glare_official_conditions.py'),
                            '--plan',cfg['source_plan']],cwd=ROOT,check=True,stdout=sys.stderr,
                           env=cluster.environment(cluster.node_by_name(cfg,'node11'),'',cfg))
            from assemble_glare_training_caches import assemble
            result=dict(assembled=True,prosody=cluster.read(Path(cfg['source_plan'])/'LATEST.json'),
                        all_training_caches=assemble(cfg,m))
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
