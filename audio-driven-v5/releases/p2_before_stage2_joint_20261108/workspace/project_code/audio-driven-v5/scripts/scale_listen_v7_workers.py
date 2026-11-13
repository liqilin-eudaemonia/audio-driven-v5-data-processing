"""Add bounded worker groups to the frozen shared queue; drain before reducing.

Each group adds one model worker per GPU and four CPU preparation slots per
worker. No in-flight source job is abandoned when a group is drained.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import run_listen_v7_fast as fast

OUT=fast.OUT
GROUPS=fast.SCHED/'extra_groups'


def worker(group,node,gpu):
    record=fast.bulk.read(GROUPS/f'group_{group}_node_{node}.json')
    assert record['code_sha256']==fast.bulk.sha(__file__)
    original_claim=fast.claim
    stop=GROUPS/f'drain_group_{group}'
    def claim(queue,rank):
        return None if stop.exists() else original_claim(queue,rank)
    fast.claim=claim
    fast.worker(group*24+node*8+gpu)


def supervisor(group,node):
    fast.verify();GROUPS.mkdir(parents=True,exist_ok=True)
    name=f'group_{group}_node_{node}'
    with (GROUPS/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=dict(group=group,node=node,state='starting',utc=fast.bulk.now(),supervisor_pid=os.getpid(),
            code_sha256=fast.bulk.sha(__file__),extra_gpu_workers=8,extra_cpu_preparation_slots=32)
        fast.bulk.write(GROUPS/(name+'.json'),record)
        children=[]
        for gpu in range(8):
            rank=group*24+node*8+gpu
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            with (OUT/'logs'/f'fast_rank_{rank:02d}.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                    '--group',str(group),'--node',str(node),'--gpu',str(gpu)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            children.append((rank,p))
        record.update(state='running',workers=[dict(rank=r,pid=p.pid) for r,p in children])
        fast.bulk.write(GROUPS/(name+'.json'),record)
        codes={str(r):p.wait() for r,p in children}
        record.update(state='finished',finished_utc=fast.bulk.now(),exit_codes=codes)
        fast.bulk.write(GROUPS/(name+'.json'),record)
        if any(codes.values()):raise SystemExit(1)


def detach(group,node):
    GROUPS.mkdir(parents=True,exist_ok=True)
    assert not (GROUPS/f'drain_group_{group}').exists(),'Group already drained; use a fresh group ID'
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'supervisor','--group',str(group),'--node',str(node)]
    with (OUT/'logs'/f'extra_group_{group}_node_{node}.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    result=dict(pid=p.pid,utc=fast.bulk.now(),group=group,node=node,command=cmd)
    fast.bulk.write(GROUPS/f'launch_group_{group}_node_{node}.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['detach','supervisor','worker','drain'])
    p.add_argument('--group',type=int,choices=range(1,8),required=True)
    p.add_argument('--node',type=int,choices=range(3),default=0);p.add_argument('--gpu',type=int,choices=range(8),default=0)
    a=p.parse_args()
    if a.stage=='detach':detach(a.group,a.node)
    elif a.stage=='supervisor':supervisor(a.group,a.node)
    elif a.stage=='worker':worker(a.group,a.node,a.gpu)
    else:
        GROUPS.mkdir(parents=True,exist_ok=True)
        (GROUPS/f'drain_group_{a.group}').write_text(fast.bulk.now()+'\n')
        print('Group will stop claiming and finish already-claimed pairs',flush=True)
