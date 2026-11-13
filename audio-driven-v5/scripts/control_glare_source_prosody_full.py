#!/usr/bin/env python3
"""Durable node11 GPU7 source-prosody task; plan wait, extraction, final sealed manifests."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import lock,save,read,sha
from prepare_glare_prosody_subset import BASE,protocol,require


def alive_plan(plan):
    launch=read(plan/'plan_launch.json');pid=launch['pid'];proc=Path('/proc')/str(pid)
    try:
        rest=(proc/'stat').read_text().rsplit(')',1)[1].split()
        argv=[p.decode() for p in (proc/'cmdline').read_bytes().split(b'\0') if p]
        return rest[0]!='Z' and str(ROOT/'scripts/prepare_glare_prosody_subset.py') in argv and '--full' in argv
    except (FileNotFoundError,ProcessLookupError):return False


def run(plan):
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    with lock(plan/'supervisor.lock',False):
        state=dict(controller_pid=os.getpid(),started=time.time(),stage='waiting_index',plan=str(plan),
                   gpu=7,controller_sha256=sha(__file__),worker_sha256=sha(ROOT/'scripts/run_glare_source_prosody_full.py'))
        def pulse():state['heartbeat']=time.time();save(plan/'controller_status.json',state)
        while not (plan/'protocol.json').exists():
            if stopped[0] or (plan/'STOP').exists():state['stage']='stopped';pulse();return
            require(alive_plan(plan),'Full index builder exited before protocol commit; inspect plan log')
            pulse();time.sleep(3)
        p=protocol(plan);require(p['requested_samples'] is None,'Controller requires explicit full source plan')
        state.update(pairs=p['pairs'],tracks=p['tracks'],chunks=p['chunks'])
        args=[sys.executable,str(ROOT/'scripts/run_glare_source_prosody_full.py'),'--plan',str(plan),'--loaders','2','--batch-size','16','--queue-size','64','--retry-execution']
        policy=read(plan/'execution_policy.json') if (plan/'execution_policy.json').exists() else {}
        state['execution_policy']=policy
        if policy.get('generation')=='official_complete':
            args[args.index('--batch-size')+1]=str(policy.get('batch_size',8));args+=['--official-resources']
        early=BASE/'capacity_probe/ACCEPTED.json'
        if policy.get('generation')!='official_complete' and early.exists():args+=['--early-reject-acceptance',str(early)]
        for attempt in range(4):
            if stopped[0] or (plan/'STOP').exists():state['stage']='stopped';pulse();return
            if policy.get('generation')=='official_complete':
                while True:
                    available=float(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())/1024
                    if available>=20:break
                    state.update(stage='waiting_gpu_capacity',free_gpu_gib=available);pulse()
                    if stopped[0] or (plan/'STOP').exists():state['stage']='stopped';pulse();return
                    time.sleep(15)
            log=(plan/'worker.log').open('a')
            proc=subprocess.Popen(args,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage='extracting',worker_pid=proc.pid,attempt=attempt+1,command=args);pulse()
            signaled=False
            while proc.poll() is None:
                if stopped[0] or (plan/'STOP').exists():
                    if not signaled:proc.terminate();signaled=True
                    state['stage']='stopping'
                pulse();time.sleep(3)
            log.close();state['worker_exit_code']=proc.returncode
            if stopped[0] or (plan/'STOP').exists():state['stage']='stopped';pulse();return
            if proc.returncode==0:break
            # Integrity/API errors are fatal. Only unexplained OS termination is restartable.
            if proc.returncode not in (-9,-11,-6,137) or attempt==3:
                raise RuntimeError('Full extraction failed; inspect worker.log; never convert a systemic error into NULL')
            state['stage']='restarting_after_process_termination';pulse();time.sleep(3)
        result=read(plan/'status.json');require(result['stage']=='complete','Worker stopped without full completion')
        state['stage']='assembling';pulse()
        with (plan/'assemble.log').open('a') as log:
            if policy.get('conditions')=='official_normalized':
                command=[sys.executable,str(ROOT/'scripts/prepare_glare_official_conditions.py'),'--plan',str(plan)]
            else:command=[sys.executable,str(ROOT/'scripts/prepare_glare_prosody_subset.py'),'--plan',str(plan),'assemble']
            subprocess.run(command,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(stage='complete',finished=time.time(),latest=read(plan/'LATEST.json'));pulse()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=BASE/'full')
    a=parser.parse_args();a.plan=a.plan.resolve()
    try:run(a.plan)
    except BaseException as exc:
        current=read(a.plan/'controller_status.json') if (a.plan/'controller_status.json').exists() else {}
        current.update(stage='failed',error=repr(exc),finished=time.time())
        target=a.plan/'controller_status.json'
        if current.get('controller_pid',os.getpid())!=os.getpid():target=a.plan/('controller_failure_'+str(os.getpid())+'.json')
        save(target,current)
        raise
