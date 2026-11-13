"""Bounded telemetry tolerance and recovery for the one Seamless local run."""
import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class Sample:
    def __init__(self, original, clock=time.monotonic, sleep=time.sleep):
        self.original=original;self.clock=clock;self.sleep=sleep;self.last=None;self.at=0

    def __call__(self, previous):
        for attempt in range(3):
            try:
                current, metrics=self.original(previous)
                self.last=copy.deepcopy(metrics);self.at=self.clock()
                return current,dict(metrics,telemetry_stale=False,telemetry_age_seconds=0.)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as error:
                age=self.clock()-self.at
                if self.last is not None and age<=60:
                    return previous,dict(copy.deepcopy(self.last),telemetry_stale=True,
                        telemetry_age_seconds=age,telemetry_error=repr(error))
                if self.last is not None or attempt==2:
                    raise RuntimeError('GPU telemetry unavailable beyond grace') from error
                self.sleep(1)


def install(cluster):
    if not isinstance(cluster.sample_node,Sample):cluster.sample_node=Sample(cluster.sample_node)


def process_probe(script):
    # pgrep narrows the process list; exact argv and status establish ownership.
    return '''import subprocess,psutil,json
script=%r
p=subprocess.run(['pgrep','-f',script],capture_output=True,text=True)
active=[]
for token in p.stdout.split():
 try:
  proc=psutil.Process(int(token));args=proc.cmdline()
  if proc.status()!=psutil.STATUS_ZOMBIE and script in args and any(x in args for x in ['worker','launch-node','supervise','supervise-existing']):active.append(proc.pid)
 except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
print(json.dumps(active))
''' % str(script)


def recoverable(runtime):
    return runtime.get('state')=='stopping_after_error' and any(s in runtime.get('error','') for s in
        ('Missing node heartbeat','stale heartbeat','GPU telemetry unavailable beyond grace'))


def allowed(attempts,now):
    return len(attempts)<6 and sum(now-t<1800 for t in attempts)<3


def enough(probes,expected_nodes=3):
    return len(probes)==expected_nodes and all(p['free_gpu_mib']>=24576 and p['memory_fraction']>=.15 for p in probes)


def probes(run):
    cfg=run.cluster.configuration()
    code="""import json,subprocess
r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
a=[int(x) for x in r.stdout.splitlines()];assert len(a)==8
m={s.split(':')[0]:int(s.split()[1]) for s in open('/proc/meminfo') if s.startswith(('MemTotal:','MemAvailable:'))}
print(json.dumps(dict(free_gpu_mib=min(a),memory_fraction=m['MemAvailable']/m['MemTotal'])))
"""
    def probe(n):
        active=json.loads(run.cluster.ssh(n,[cfg['python'],'-c',process_probe(run.SCRIPT)]))
        if active:return dict(node=n['name'],active=active,free_gpu_mib=0,memory_fraction=0)
        r=json.loads(run.cluster.ssh(n,[cfg['python'],'-c',code]));r.update(node=n['name'],active=[]);return r
    with concurrent.futures.ThreadPoolExecutor(len(cfg['nodes'])) as pool:return list(pool.map(probe,cfg['nodes']))


def watch(run):
    run.verify();run.bind();out=run.SCHED/'watchdog';out.mkdir(exist_ok=True)
    with (out/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=run.read(out/'attempts.json') if (out/'attempts.json').exists() else []
        while True:
            try:
                r=run.read(run.OUT/'runtime.json');now=time.time()
                status=dict(utc=run.full.now(),pid=os.getpid(),state='monitoring',run_state=r['state'],attempts=len(attempts))
                if r['state'] in ('finished','finished_with_errors'):
                    run.write(out/'status.json',dict(status,state='finished'));return
                if recoverable(r):
                    if not allowed(attempts,now):status['state']='recovery_limit_reached'
                    else:
                        resources=probes(run);status['resources']=resources
                        if any(x['active'] for x in resources):status['state']='waiting_for_run_processes_to_exit'
                        elif not enough(resources,len(run.cluster.configuration()['nodes'])):status['state']='waiting_for_resource_headroom'
                        else:
                            run.verify();attempts.append(now);run.write(out/'attempts.json',attempts)
                            run.write(out/'status.json',dict(status,state='recovering'))
                            run.resume();status['state']='restart_requested'
                elif r['state']=='stopping_after_error':status.update(state='requires_inspection',error=r.get('error'))
                run.write(out/'status.json',status)
            except Exception as e:
                run.write(out/'status.json',dict(utc=run.full.now(),pid=os.getpid(),state='monitor_error',error=repr(e)))
            time.sleep(30)


def detach(run):
    run.verify();out=run.SCHED/'watchdog';out.mkdir(exist_ok=True)
    # Check the same lock before creating another monitor.
    with (out/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (out/'watch.log').open('ab',buffering=0) as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'watch'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    run.write(out/'launch.json',dict(utc=run.full.now(),pid=p.pid,maximum_total_attempts=6,maximum_per_30_minutes=3))
    print(json.dumps(dict(pid=p.pid)),flush=True)


if __name__=='__main__':
    import run_listen_local_full as run
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['watch','detach','probe']);a=p.parse_args()
    if a.stage=='watch':watch(run)
    elif a.stage=='detach':detach(run)
    else:run.bind();print(json.dumps(probes(run)),flush=True)
