"""Monitor this one authorized run; bounded recovery of transient scheduler stops."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_listen_v8_audio_governor as governor
run=governor.run
OUT=run.SCHED/'watchdog'
SCRIPT=Path(__file__).resolve()
RECOVERABLE=('Missing node heartbeat','stale heartbeat','persistent critical memory pressure')


def recoverable(runtime):
    return runtime.get('state')=='stopping_after_error' and any(
        message in runtime.get('error','') for message in RECOVERABLE)


def ready_to_restart(probes):
    # Reserve enough room to reload all four existing model processes per GPU.
    return len(probes)==3 and all(p['free_gpu_mib']>=30720 and p['memory_fraction']>=.15 for p in probes)


def resources():
    code="""import subprocess,json
p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
rows=[list(map(int,s.split(','))) for s in p.stdout.strip().splitlines()]
mem={s.split(':')[0]:int(s.split()[1]) for s in open('/proc/meminfo') if s.startswith(('MemTotal:','MemAvailable:'))}
assert len(rows)==8
print(json.dumps(dict(free_gpu_mib=min(b-a for a,b in rows),memory_fraction=mem['MemAvailable']/mem['MemTotal'])))
"""
    cfg=run.cluster.configuration()
    def probe(node):
        return json.loads(run.cluster.ssh(node,[cfg['python'],'-c',code]))
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        return list(pool.map(probe,cfg['nodes']))


def restart():
    governor.verify()
    # resume() independently checks the actual processes on all nodes before
    # resetting a cursor. It preserves all committed results and source contracts.
    run.resume()
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        runtime=run.cluster.read(run.OUT/'runtime.json')
        try:
            nodes=[run.cluster.read(run.SCHED/'nodes'/(n['name']+'.json')) for n in run.cluster.configuration()['nodes']]
        except (FileNotFoundError,json.JSONDecodeError):
            nodes=[]
        if runtime.get('state')=='cluster_screening' and len(nodes)==3 and all(
                n['state']=='running' and time.time()-n['unix']<45 for n in nodes):
            governor.handoff()
            break
        time.sleep(2)
    else:raise RuntimeError('Recovery node startup timed out')
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        runtime=run.cluster.read(run.OUT/'runtime.json')
        if runtime.get('governor_revision')==governor.POLICY['revision'] and runtime.get('state')=='cluster_screening':
            ctrl=run.cluster.read(run.SCHED/'control.json')
            assert not ctrl['stop']
            ctrl['nodes']={name:run.cluster.dimensions(4,2) for name in ctrl['nodes']}
            ctrl['stage']='recovered_measured_4x2'
            run.write(run.SCHED/'control.json',ctrl)
            return
        time.sleep(1)
    raise RuntimeError('Recovery coordinator startup timed out')


def watch():
    governor.verify();run.bind();OUT.mkdir(exist_ok=True)
    with (OUT/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=run.read(OUT/'attempts.json') if (OUT/'attempts.json').exists() else []
        completed=0
        while True:
            try:
                runtime=run.cluster.read(run.OUT/'runtime.json');now=time.time()
                completed=max(completed,len(list((run.OUT/'summaries').glob('*.json'))))
                status=dict(utc=run.full.now(),pid=os.getpid(),state='monitoring',run_state=runtime['state'],
                    complete_pairs_lower_bound=completed,scope='One authorized run only; never terminate unrelated tasks')
                if runtime['state'] in ('finished','finished_with_errors'):
                    status['state']='finished';run.write(OUT/'status.json',status);return
                if recoverable(runtime):
                    nodes=[run.cluster.read(run.SCHED/'nodes'/(n['name']+'.json')) for n in run.cluster.configuration()['nodes']]
                    quiet=all(n['state'] in ('finished','failed') and
                        all(c['exit_code'] is not None for c in n['children'].values()) for n in nodes)
                    recent=[t for t in attempts if now-t<1800]
                    if not quiet:status['state']='waiting_for_run_workers_to_drain'
                    elif len(recent)>=3:status['state']='recovery_rate_limited'
                    else:
                        probes=resources();status['resources']=probes
                        if ready_to_restart(probes):
                            attempts.append(now);run.write(OUT/'attempts.json',attempts)
                            status['state']='recovering';run.write(OUT/'status.json',status)
                            restart();status['state']='recovered'
                        else:status['state']='waiting_for_resource_headroom'
                elif runtime['state']=='stopping_after_error':
                    status.update(state='requires_inspection',error=runtime.get('error'))
                run.write(OUT/'status.json',status)
            except Exception as e:
                run.write(OUT/'status.json',dict(utc=run.full.now(),state='monitor_error',error=repr(e),pid=os.getpid()))
            time.sleep(30)


def detach():
    OUT.mkdir(exist_ok=True)
    with (OUT/'watch.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'watch'],stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,start_new_session=True)
    run.write(OUT/'launch.json',dict(utc=run.full.now(),pid=p.pid,code_sha256=run.sha(SCRIPT),
        recovery_scope=list(RECOVERABLE),maximum_attempts_per_30_minutes=3))
    print(json.dumps(run.read(OUT/'launch.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['watch','detach']);args=parser.parse_args()
    if args.stage=='watch':watch()
    else:detach()
