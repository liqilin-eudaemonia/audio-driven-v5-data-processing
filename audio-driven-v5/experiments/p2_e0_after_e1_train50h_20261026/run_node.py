#!/usr/bin/env python3
"""Detached node supervisor; fresh E0 initialization, same immutable E1 config."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from control import OUT,ROOT,TRAIN,PYTHON,SEED,HOSTNAMES,read,write,now,verify_release


def process_birth(pid):
    # fields after final ')' start at proc stat field 3; starttime is field 22.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def gpu_state():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[tuple(map(int,line.split(','))) for line in output.strip().splitlines()]
    return dict(gpus=[dict(index=i,memory_mib=m) for i,m in rows],idle=len(rows)==8 and [i for i,m in rows]==list(range(8)) and max(m for i,m in rows)<1024)


def command(rank,resume=None):
    cmd=[str(PYTHON),'-u','-m','torch.distributed.run','--nnodes=3','--nproc-per-node=8',
         f'--node-rank={rank}','--master-addr=172.29.252.11','--master-port=29747','--max-restarts=0',
         '--log-dir',str(OUT/'worker_logs'/f'node{rank}'),'--tee=3','scripts/p2_listen.py','train',
         '--config',str(TRAIN/'config.json'),'--arm','E0','--train-seed',str(SEED)]
    if resume:cmd+=['--resume',str(resume)]
    return cmd


def main():
    p=argparse.ArgumentParser();p.add_argument('--node-rank',type=int,choices=range(3),required=True)
    p.add_argument('--detach',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--stop-owned',action='store_true')
    p.add_argument('--resume',type=Path)
    a=p.parse_args();status_path=OUT/f'node{a.node_rank}.json'
    if socket.gethostname()!=HOSTNAMES[a.node_rank]:raise RuntimeError('Node rank/hostname mismatch')
    if a.stop_owned:
        if not status_path.exists():return
        state=read(status_path)
        if state.get('state') not in ('running','starting'):return
        pid=state.get('torchrun_pid')
        if not pid or not Path(f'/proc/{pid}').exists():return
        if process_birth(pid)!=state['torchrun_starttime'] or os.getpgid(pid)!=pid:raise RuntimeError('PID ownership changed')
        tokens=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'E0' not in tokens or str(TRAIN/'config.json').encode() not in tokens:raise RuntimeError('Refuse to stop unrelated process')
        os.killpg(pid,signal.SIGTERM);return
    verify_release()
    if a.probe:
        print(json.dumps(dict(hostname=socket.gethostname(),node_rank=a.node_rank,source_verified=True,**gpu_state())));return
    if a.detach:
        if status_path.exists() and read(status_path).get('state') in ('starting','running'):raise RuntimeError('Node already active')
        cmd=[str(PYTHON),str(Path(__file__).resolve()),'--node-rank',str(a.node_rank)]
        if a.resume:cmd+=['--resume',str(a.resume.resolve())]
        with (OUT/f'node{a.node_rank}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(150):
            if status_path.exists():
                state=read(status_path)
                if state.get('supervisor_pid')==child.pid and state['state'] in ('running','failed','finished'):
                    print(json.dumps(state));sys.exit(0 if state['state']=='running' else 1)
            if child.poll() is not None:raise RuntimeError(f'Node supervisor exited {child.returncode}; see log')
            time.sleep(.1)
        raise TimeoutError('Node supervisor did not acknowledge launch')
    with (OUT/f'node{a.node_rank}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(state='starting',node_rank=a.node_rank,hostname=socket.gethostname(),supervisor_pid=os.getpid(),started_at=now())
        write(status_path,state)
        child=None
        try:
            gpu=gpu_state()
            if not gpu['idle']:raise RuntimeError('Expected all eight GPUs idle: '+str(gpu))
            if (TRAIN/f'runs/{SEED}/E0/run.json').exists() and a.resume is None:raise RuntimeError('E0 exists; explicit resume required')
            cmd=command(a.node_rank,a.resume.resolve() if a.resume else None)
            script='source '+shlex.quote(str(ROOT/'configs/p2_listen_h200_multinode.env'))+'\nexec '+shlex.join(cmd)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONUNBUFFERED='1')
            child=subprocess.Popen(['/bin/bash','-c',script],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,start_new_session=True)
            state.update(state='running',torchrun_pid=child.pid,torchrun_starttime=process_birth(child.pid),command=cmd,
                         resume=str(a.resume) if a.resume else None,source_verified=True)
            write(status_path,state);print(json.dumps(state),flush=True)
            rc=child.wait()
            state.update(state='finished' if rc==0 else 'failed',returncode=rc,finished_at=now())
            write(status_path,state);sys.exit(rc)
        except Exception as error:
            if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
            state.update(state='failed',returncode=1,error=str(error),finished_at=now());write(status_path,state);raise


if __name__=='__main__':main()
