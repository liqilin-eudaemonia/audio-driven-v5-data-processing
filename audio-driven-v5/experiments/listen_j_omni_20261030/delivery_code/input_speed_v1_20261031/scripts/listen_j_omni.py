#!/usr/bin/env python3
"""Launch persistent, resumable J-pair natural captions after nod completion."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,acquire,JobQueue
from listen_benchmark.j_omni import producer,producer_step,worker
from listen_j_annotation_supervisor import identity,alive,spawn
PYTHON='/private/software/conda/envs/vllm/bin/python'


def environment(gpu=None):
    out=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    if gpu is not None:out['CUDA_VISIBLE_DEVICES']=str(gpu)
    return out


def complete(work):
    if not (work/'producer_complete.json').exists():return False
    q=JobQueue(work,'caption');s=read(q.path)
    return not s['inflight'] and not s.get('deferred') and s['offset']>=(work/'ready.jsonl').stat().st_size


def supervise(work,node):
    cfg=read(work/'config.json');nc=cfg['nodes'][node]
    if socket.gethostname()!=nc['hostname']:raise ValueError('Supervisor host mismatch')
    nodework=work/'nodes'/node;owner=acquire(nodework/'supervisor.lock',False)
    if owner is None:return
    sf=nodework/'supervisor.json';state=read(sf) if sf.exists() else {'children':{},'restarts':{}}
    state.update(pid=os.getpid(),start_ticks=identity(os.getpid()),started_at=time.time(),node=node)
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping)
    while not stop:
        try:
            if complete(work):state['stage']='complete_with_issues_possible';break
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
            devices={int(x.split(',')[0]):list(map(float,x.split(',')[1:])) for x in raw.strip().splitlines()}
            producer_state=read(work/'producer.json') if (work/'producer.json').exists() else {}
            if not (work/'producer_complete.json').exists() and time.time()-producer_state.get('at',0)>45:
                child=state['children'].get('producer')
                if not alive(child):state['children']['producer']=spawn(nodework,'producer',[PYTHON,str(Path(__file__).resolve()),'producer','--work',str(work)],environment())
            for gpu in nc['gpu_ids']:
                key=f'gpu-{gpu}';child=state['children'].get(key)
                if not alive(child):
                    if devices[gpu][0]>2000:continue
                    restarts=state['restarts'].get(key,[]);restarts=[t for t in restarts if time.time()-t<3600]
                    if len(restarts)>=4:state.setdefault('blocked_gpus',{})[str(gpu)]='four exits in one hour; inspect logs';continue
                    state['children'][key]=spawn(nodework,key,[PYTHON,str(Path(__file__).resolve()),'worker','--work',str(work),'--node',node,'--gpu',str(gpu)],environment(gpu))
                    state['restarts'][key]=restarts+[time.time()]
            q=JobQueue(work,'caption');qs=read(q.path)
            state.update(stage='running',devices=devices,queue=dict(resolved_clips=qs['completed_clips'],completed_sources=qs['completed'],failed_sources=len(qs['failed']),inflight_sources=len(qs['inflight'])),error=None)
        except Exception as e:state['error']=repr(e)
        state['heartbeat']=time.time();write(sf,state)
        time.sleep(15)
    state.update(stopped_at=time.time(),heartbeat=time.time());write(sf,state);owner.close()


def watchdog(work,node):
    nodework=work/'nodes'/node;owner=acquire(nodework/'watchdog.lock',False)
    if owner is None:return
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping)
    state=dict(pid=os.getpid(),start_ticks=identity(os.getpid()),started_at=time.time(),restarts=[])
    while not stop:
        try:
            if (nodework/'STOP_SUPERVISION').exists() or complete(work):break
            f=nodework/'supervisor.json';s=read(f) if f.exists() else None
            if not alive(s):
                state['restarts'].append(spawn(nodework,'supervisor',[PYTHON,str(Path(__file__).resolve()),'supervise','--work',str(work),'--node',node],environment()))
                state['restarts']=state['restarts'][-20:]
            state['error']=None
        except Exception as e:state['error']=repr(e)
        state['heartbeat']=time.time();write(nodework/'watchdog.json',state);time.sleep(15)
    state['stopped_at']=time.time();write(nodework/'watchdog.json',state);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['producer','produce-once','worker','supervise','watchdog'])
    p.add_argument('--work',type=Path,required=True);p.add_argument('--node');p.add_argument('--gpu',type=int);p.add_argument('--max-clips',type=int)
    a=p.parse_args();w=a.work.resolve()
    if a.command=='producer':producer(w)
    elif a.command=='produce-once':print(json.dumps(producer_step(w)))
    elif a.command=='worker':worker(w,a.node,a.gpu,a.max_clips)
    elif a.command=='supervise':supervise(w,a.node)
    else:watchdog(w,a.node)
