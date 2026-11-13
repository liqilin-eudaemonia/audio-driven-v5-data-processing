#!/usr/bin/env python3
"""Explicitly launched four-arm test; wait for assigned GPUs, preserve other jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster
from listen_cluster import transport
from run_glare_cache_cluster import status


def resources(cfg):
    ssh=cluster.read(cfg['ssh_config'])
    def probe(node):
        args=['nvidia-smi','-i',','.join(map(str,node['gpus'])),
              '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
        r=subprocess.run(transport(ssh,node,args),capture_output=True,text=True,timeout=30)
        rows=[]
        if r.returncode==0:
            rows=[dict(gpu=int(v[0]),memory_mib=int(v[1]),utilization=int(v[2]))
                  for line in r.stdout.splitlines() if (v:=line.split(','))]
        return dict(node=node['name'],gpus=rows,error=r.stderr[-1000:] if r.returncode else None,
                    free=r.returncode==0 and len(rows)==len(node['gpus']) and
                    all(x['memory_mib']<=cfg['preflight']['max_used_gpu_memory_mib'] for x in rows))
    with ThreadPoolExecutor(max_workers=2) as pool:return list(pool.map(probe,cfg['nodes']))


def command(config,action,log):
    cfg=cluster.config(config)
    with Path(log).open('a') as stream:
        return subprocess.run([cfg['python'],str(ROOT/'scripts/run_glare_cache_cluster.py'),
                               '--config',str(config),action],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode


def metrics(cfg,m):
    result={}
    for stage in cluster.STAGES:
        q=cluster.TaskQueue(cfg['work'],m['run_id'],stage);rows=[]
        for task in q.tasks:
            path=q.folder(task)/'done.json'
            if path.exists():rows.append(cluster.read(path))
        native=[r['result']['native_state'] for r in rows]
        first=min((s['started'] for s in native),default=None)
        last=max((r['finished'] for r in rows),default=None)
        units=sum(r['task']['units'] for r in rows)
        result[stage]=dict(tasks=len(rows),units=units,first_native_start=first,last_commit=last,
             active_window_seconds=(last-first) if first and last else None,
             units_per_second=units/(last-first) if first and last and last>first else None,
             native_gpu_or_encode_seconds=sum(s.get('inference_s',s.get('encode_s',0)) for s in native),
             writer_seconds=sum(s.get('async_writes',{}).get('work_s',0) for s in native),
             writer_backpressure_seconds=sum(s.get('async_writes',{}).get('backpressure_s',0) for s in native))
    return result


def run(plan_path,max_minutes):
    plan=cluster.read(plan_path);folder=plan_path.parent
    stop=folder/'SUITE_STOP';state_path=folder/'SUITE_STATUS.json'
    interrupted=[False];signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:interrupted.__setitem__(0,True))
    state=dict(pid=os.getpid(),started=time.time(),stage='waiting_resources',arms=[],active=None,
               physical_gpus=28,node11_gpus=[0,1,2,3],other_jobs_modified=False)
    def stopping():return interrupted[0] or stop.exists()
    def pulse():state['heartbeat']=time.time();cluster.save(state_path,state)
    with cluster.locked(folder/'suite.lock'):
        try:
            for arm in plan['arms']:
                if stopping():break
                config=Path(arm['config']);cfg=cluster.config(config);m=cluster.load_run(cfg)
                state.update(active=arm['name'],stage='waiting_resources');pulse()
                free=0
                while free<2 and not stopping():
                    probes=resources(cfg);state['resources']=probes
                    free=free+1 if all(n['free'] for n in probes) else 0
                    pulse()
                    if free<2:time.sleep(5)
                if stopping():break
                # No writer from the preceding arm may overlap the next one.
                with cluster.writer_fences(cfg,m,shared=False):pass
                state.update(stage='launching',launch_requested=time.time());pulse()
                log=folder/(arm['name']+'.suite.log')
                rc=command(config,'start',log)
                if rc:raise RuntimeError('Arm launch failed: '+arm['name']+'; see '+str(log))
                state.update(stage='running',arm_started=time.time());pulse()
                while not stopping():
                    value=status(cfg,m)
                    state['queue']=value['queue'];state['nodes']={n:v.get('stage') for n,v in value['nodes'].items()}
                    state['resources']=resources(cfg);pulse()
                    if any(v.get('stage')=='failed' for v in value['nodes'].values()):
                        raise RuntimeError('Node supervisor failed in '+arm['name'])
                    if all(q['exhausted'] for q in value['queue'].values()):
                        if not value['complete']:raise RuntimeError('Input task retries exhausted in '+arm['name'])
                        if all(n.get('stage')=='complete' for n in value['nodes'].values()):break
                    if time.time()-state['arm_started']>max_minutes*60:raise TimeoutError('Benchmark arm exceeded time limit')
                    time.sleep(10)
                if stopping():command(config,'stop',log);break
                result=dict(name=arm['name'],run_id=m['run_id'],metrics=metrics(cfg,m),
                            finished=time.time(),config=str(config),complete=True)
                cluster.save(folder/(arm['name']+'.RESULT.json'),result);state['arms'].append(result);pulse()
                # Node status is written just before its process exits; allow its
                # shared fences to be released before proceeding to another run.
                deadline=time.time()+60
                while True:
                    try:
                        with cluster.writer_fences(cfg,m,shared=False):pass
                        break
                    except BlockingIOError:
                        if time.time()>deadline:raise RuntimeError('Prior arm did not release cache fences')
                        time.sleep(2)
            state.update(stage='stopped' if stopping() else 'complete',finished=time.time());pulse()
            if not stopping():cluster.save(folder/'SUITE_RESULTS.json',state)
        except BaseException as exc:
            state.update(stage='failed',error=repr(exc),traceback=traceback.format_exc(),finished=time.time());pulse()
            if state.get('active'):
                arm=next(a for a in plan['arms'] if a['name']==state['active'])
                command(Path(arm['config']),'stop',folder/(state['active']+'.suite.log'))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=ROOT/'experiments/glare_cache_28gpu_20261110/benchmark_ready/TEST_PLAN.json')
    p.add_argument('--max-minutes',type=int,default=30);p.add_argument('--detach',action='store_true')
    a=p.parse_args();a.plan=a.plan.resolve()
    if a.detach:
        # This command is an explicit new launch request, not an automatic resume.
        (a.plan.parent/'SUITE_STOP').unlink(missing_ok=True)
        with (a.plan.parent/'suite.log').open('a') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--plan',str(a.plan),
                                      '--max-minutes',str(a.max_minutes)],cwd=ROOT,stdin=subprocess.DEVNULL,
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=process.pid,plan=str(a.plan),launched=time.time())
        cluster.save(a.plan.parent/'SUITE_LAUNCH.json',info);print(json.dumps(info))
    else:run(a.plan,a.max_minutes)
