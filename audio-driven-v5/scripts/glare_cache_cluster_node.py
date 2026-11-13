#!/usr/bin/env python3
"""Detached node supervisor. Owns assigned children only; never calls pkill."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster


def run(config_path,node_name):
    cfg=cluster.config(config_path);m=cluster.load_run(cfg);node=cluster.node_by_name(cfg,node_name)
    cluster.assert_local(node);work=Path(cfg['work']);folder=work/'nodes'/node_name
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    with cluster.locked(folder/'supervisor.lock'),cluster.writer_fences(cfg,m):
        cluster.assert_active(cfg,m)
        cluster.validate_node_inputs(m,cluster.read(folder/'preflight.json'))
        cluster.verify_assets(m,folder/'assets.json')
        count=cluster.execution(cfg)['lia_workers_per_gpu']
        keys=[(g,s) for g in node['gpus'] for s in range(count)]
        label=lambda key:'%d/%d'%key
        children={};failures={label(k):0 for k in keys};last_stage={};last_mode={}
        queues={s:cluster.TaskQueue(work,m['run_id'],s) for s in cluster.STAGES}
        state=dict(node=node_name,host=node['host'],run_id=m['run_id'],pid=os.getpid(),started=time.time(),stage='running')
        def spawn(gpu,worker_slot,stage):
            key=(gpu,worker_slot);slot=cluster.slot_dir(cfg,node,gpu,worker_slot);slot.mkdir(parents=True,exist_ok=True)
            cmd=[cfg['python'],str(ROOT/'scripts/glare_cache_cluster_worker.py'),'--config',str(config_path),
                 '--node',node_name,'--gpu',str(gpu),'--stage',stage,'--worker-slot',str(worker_slot)]
            log=(slot/'worker.log').open('a')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=cluster.environment(node,gpu,cfg),stdout=log,stderr=subprocess.STDOUT)
            log.close();children[key]=proc;last_stage[key]=stage;last_mode[gpu]=stage
            cluster.save(slot/'launch.json',dict(pid=proc.pid,stage=stage,command=cmd,started=time.time(),run_id=m['run_id']))
        try:
            while True:
                stop=stopping[0] or (work/'STOP').exists()
                summaries={s:q.summary() for s,q in queues.items()}
                for key,proc in list(children.items()):
                    tag=label(key)
                    if proc is not None and proc.poll() is None:
                        if stop and tag not in state.get('signaled',[]):
                            proc.terminate();state.setdefault('signaled',[]).append(tag)
                        continue
                    if proc.returncode!=0 and not stop:failures[tag]+=1
                    del children[key]
                for gpu in node['gpus']:
                    if stop or all(s['exhausted'] for s in summaries.values()):continue
                    active=[last_stage[k] for k in children if k[0]==gpu]
                    preferred='prosody' if gpu in node['prosody_preferred_gpus'] else 'lia'
                    previous=last_mode.get(gpu)
                    stage=active[0] if active else (('lia' if previous=='prosody' else 'prosody') if previous else preferred)
                    assert not active or len(set(active))==1,'Never co-load Qwen and LIA workers on the same GPU'
                    if summaries[stage]['exhausted']:stage='lia' if stage=='prosody' else 'prosody'
                    # An existing peer drains before a physical GPU changes phase.
                    if active and stage!=active[0]:continue
                    for worker_slot in range(count if stage=='lia' else 1):
                        key=(gpu,worker_slot)
                        if key not in children and failures[label(key)]<3:spawn(gpu,worker_slot,stage)
                state.update(heartbeat=time.time(),failures=failures,queue=summaries,
                             workers={label(k):dict(pid=p.pid,gpu=k[0],worker_slot=k[1],returncode=p.poll(),stage=last_stage[k]) for k,p in children.items()})
                if not children:
                    state['stage']=('stopped' if stop else 'complete' if all(s['complete'] for s in summaries.values())
                                    else 'finished_with_errors' if all(s['exhausted'] for s in summaries.values()) else 'failed')
                    cluster.save(folder/'status.json',state);return
                cluster.save(folder/'status.json',state)
                time.sleep(10)
        finally:
            # Child signal handlers drain bounded batches and commit receipts.
            for p in children.values():
                if p.poll() is None:p.terminate()
            for p in children.values():
                try:p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    # Preserve ownership fences until this child really exits.
                    p.kill();p.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=cluster.DEFAULT);parser.add_argument('--node',required=True)
    args=parser.parse_args()
    run(args.config.resolve(),args.node)
