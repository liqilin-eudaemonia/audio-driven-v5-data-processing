"""Replace only the live coordinator: sustained-pressure, throughput-aware tuning.

Workers, scientific contracts, queues and committed results are unchanged.
No worker restart and no intervention in unrelated processes is performed.
"""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_listen_v8_audio_full as run

SCRIPT = Path(__file__).resolve()
DIRECTORY = run.SCHED/'governor'
POLICY = dict(revision='sustained-pressure-governor-1', maximum_models_per_gpu=4,
    maximum_prep_per_worker=2, cpu_sustained_percent=92., cpu_sustained_seconds=60.,
    cpu_critical_percent=97., cpu_critical_seconds=30., io_sustained_percent=20.,
    minimum_memory_fraction=.10, minimum_free_gpu_mib=16384,
    recovery_cpu_percent=82., recovery_free_gpu_mib=24576, recovery_memory_fraction=.15,
    change_cooldown_seconds=120., recovery_cooldown_seconds=180.,
    evaluation_seconds=240., evaluation_warmup_seconds=60., minimum_throughput_gain=.05,
    pressure_stop_seconds=120., heartbeat_timeout_seconds=120.)


def dimensions(models, prep):
    assert models in (1,2,4) and prep in (1,2)
    return run.cluster.dimensions(models,prep)


class Governor:
    def __init__(self):
        self.history = collections.defaultdict(list)
        self.changed = {}
        self.memory_pressure_since = {}
        self.evaluations = {}
        self.completed_trials = []
        self.prep_trials_disabled = set()

    @staticmethod
    def window(history, now, seconds):
        rows=[r for r in history if now-seconds <= r['unix'] <= now]
        if len(rows)<3 or rows[-1]['unix']-rows[0]['unix'] < seconds*.75:
            return []
        return rows

    def consider(self, name, node, setting, counts, now):
        history=self.history[name]
        if history and node['unix'] <= history[-1]['unix']:
            return None, []
        history.append(node)
        self.history[name]=history=[r for r in history if now-r['unix']<=360]
        models,prep=setting['models_per_gpu'],setting['prep_per_worker']
        free_gpu=min(g['total_mib']-g['memory_mib'] for g in node['gpus'])
        free_ram=node['memory']['MemAvailable']/node['memory']['MemTotal']
        memory_reasons=[]
        if free_gpu<POLICY['minimum_free_gpu_mib']:memory_reasons.append('gpu_memory')
        if free_ram<POLICY['minimum_memory_fraction']:memory_reasons.append('ram')
        if memory_reasons:
            self.memory_pressure_since.setdefault(name,now)
            if now-self.memory_pressure_since[name]>=POLICY['pressure_stop_seconds']:
                raise RuntimeError(name+' persistent critical memory pressure; drain run-owned workers')
            if (models,prep)!=(1,1):
                self.changed[name]=now
                return dimensions(1,1),memory_reasons
            return None,memory_reasons
        self.memory_pressure_since.pop(name,None)
        critical=self.window(history,now,POLICY['cpu_critical_seconds'])
        sustained=self.window(history,now,POLICY['cpu_sustained_seconds'])
        reasons=[]
        if critical and all(r['cpu_busy_percent']>=POLICY['cpu_critical_percent'] for r in critical):
            reasons.append('critical_cpu_30s')
        if sustained:
            if sum(r['cpu_busy_percent'] for r in sustained)/len(sustained)>POLICY['cpu_sustained_percent']:
                reasons.append('sustained_cpu_60s')
            if sum(r['iowait_percent'] for r in sustained)/len(sustained)>POLICY['io_sustained_percent']:
                reasons.append('sustained_io_60s')
        age=now-self.changed.get(name,-1e10)
        if reasons and (age>=POLICY['change_cooldown_seconds'] or 'critical_cpu_30s' in reasons):
            desired=dimensions(models,1) if prep>1 else dimensions(max(1,models//2),1)
            if desired!=setting:
                self.changed[name]=now
                self.evaluations.pop(name,None)
                if prep>1:self.prep_trials_disabled.add(name)
                return desired,reasons
            return None,reasons
        # Pair rates are observational on randomized disjoint work, not a matched-input benchmark.
        ev=self.evaluations.get(name)
        if ev and (models,prep)==tuple(ev['setting']):
            if ev.get('baseline') is None and now-ev['started']>=POLICY['evaluation_warmup_seconds']:
                ev.update(baseline=counts,time_baseline=now)
            elif ev.get('baseline') is not None and now-ev['time_baseline']>=POLICY['evaluation_seconds']:
                rate=(counts-ev['baseline'])*60/(now-ev['time_baseline'])
                trial=dict(node=name,setting=[models,prep],pairs_per_minute=rate,
                           elapsed=now-ev['time_baseline'],unix=now)
                self.completed_trials.append(trial)
                previous=ev.get('previous_rate')
                self.evaluations.pop(name,None)
                if prep==2 and previous is not None and rate<previous*(1+POLICY['minimum_throughput_gain']):
                    self.prep_trials_disabled.add(name);self.changed[name]=now
                    return dimensions(models,1),['prefetch_trial_no_observed_gain']
        if (models,prep)==(4,1) and name not in self.evaluations and name not in self.prep_trials_disabled:
            self.evaluations[name]=dict(setting=[4,1],started=now,baseline=None)
        stable=self.window(history,now,120.)
        can_raise=(not reasons and stable and age>=POLICY['recovery_cooldown_seconds']
            and sum(r['cpu_busy_percent'] for r in stable)/len(stable)<POLICY['recovery_cpu_percent']
            and max(r['iowait_percent'] for r in stable)<10
            and free_gpu>=POLICY['recovery_free_gpu_mib'] and free_ram>=POLICY['recovery_memory_fraction'])
        if can_raise and models<4:
            self.changed[name]=now
            return dimensions(models*2,1),['sustained_headroom_restore']
        # Increase preparation only after measuring the 4x1 baseline.
        baselines=[v for v in self.completed_trials if v['node']==name and v['setting']==[4,1]]
        if can_raise and (models,prep)==(4,1) and baselines and name not in self.prep_trials_disabled:
            self.evaluations[name]=dict(setting=[4,2],started=now,baseline=None,previous_rate=baselines[-1]['pairs_per_minute'])
            self.changed[name]=now
            return dimensions(4,2),['bounded_prefetch_throughput_trial']
        return None,[]


def node_counts():
    counts=collections.Counter()
    for p in (run.SCHED/'progress').glob('*.json'):
        value=run.cluster.read(p)
        counts[value['node']]+=value['counts'].get('complete_pairs',0)
    return counts


def verify():
    run.verify()
    contract=run.read(DIRECTORY/'contract.json')
    assert contract['code_sha256']==run.sha(SCRIPT)
    assert contract['quality_contract_sha256']==run.sha(run.OUT/'contract.json')
    assert contract['policy']==POLICY


def supervise():
    verify();run.bind()
    governor=Governor()
    reporter=None;last_report=0
    with (run.SCHED/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            ctrl=run.cluster.read(run.SCHED/'control.json')
            assert not ctrl['stop'] and not ctrl['drain']
            # Existing 32 children per node are already loaded and idle-enabled by control.
            for name,setting in ctrl['nodes'].items():
                node=run.cluster.read(run.SCHED/'nodes'/(name+'.json'))
                assert time.time()-node['unix']<60 and node['state']=='running'
                headroom=(node['memory']['MemAvailable']/node['memory']['MemTotal']>.15 and
                    min(g['total_mib']-g['memory_mib'] for g in node['gpus'])>24576 and node['cpu_busy_percent']<85)
                if headroom:
                    ctrl['nodes'][name]=dimensions(4,1);governor.changed[name]=time.time()
                    governor.evaluations[name]=dict(setting=[4,1],started=time.time(),baseline=None)
            ctrl['stage']='sustained_pressure_governor';run.write(run.SCHED/'control.json',ctrl)
            while True:
                snap=run.cluster.progress();nodes=snap['nodes'];now=time.time()
                if len(nodes)==3 and all(n['state']=='finished' for n in nodes.values()):break
                assert len(nodes)==3,'Missing node heartbeat'
                counts=node_counts();ctrl=run.cluster.read(run.SCHED/'control.json');changed=False
                for name,node in nodes.items():
                    if node['state']=='failed':raise RuntimeError(name+' worker failure')
                    if now-node['unix']>POLICY['heartbeat_timeout_seconds']:raise RuntimeError(name+' stale heartbeat')
                    desired,reasons=governor.consider(name,node,ctrl['nodes'][name],counts[name],now)
                    if desired:
                        event=dict(utc=run.full.now(),node=name,previous=ctrl['nodes'][name],setting=desired,reasons=reasons)
                        ctrl['nodes'][name]=desired;changed=True
                        with (DIRECTORY/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                if changed:run.write(run.SCHED/'control.json',ctrl)
                run.write(DIRECTORY/'trials.json',dict(utc=run.full.now(),completed=governor.completed_trials,
                    in_progress=governor.evaluations,prep_trials_disabled=sorted(governor.prep_trials_disabled)))
                if reporter is not None and reporter.poll() not in (None,0):raise RuntimeError('Statistics reporter failed')
                if now-last_report>300 and (reporter is None or reporter.poll() is not None):
                    with (run.OUT/'logs/report.log').open('ab',buffering=0) as log:
                        reporter=subprocess.Popen([sys.executable,'-u',str(run.SCRIPT),'report'],stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                    last_report=now
                run.write(run.OUT/'runtime.json',dict(state='cluster_screening',utc=run.full.now(),coordinator_pid=os.getpid(),
                    coordinator_script=str(SCRIPT),progress=snap['counts'],queue=snap['queue'],
                    nodes={k:n['state'] for k,n in nodes.items()},control=ctrl,governor_revision=POLICY['revision'],
                    visual_filter_applied=False,training_started=False))
                run.write(run.SCHED/'latest.json',snap)
                time.sleep(10)
            if reporter:reporter.wait()
            for path in (run.OUT/'results').glob('*.json'):
                if not (run.OUT/'summaries'/path.name).exists():run.write_pair_summary(run.read(path))
            summary=run.report(export=True)
            assert summary['pending_pairs']==0,'Unaccounted claims; resume required'
            run.write(run.OUT/'runtime.json',dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',
                utc=run.full.now(),summary=summary,workers=0,visual_filter_applied=False,training_started=False))
        except BaseException as e:
            ctrl=run.cluster.read(run.SCHED/'control.json');ctrl['stop']=True;run.write(run.SCHED/'control.json',ctrl)
            run.write(run.OUT/'runtime.json',dict(state='stopping_after_error',utc=run.full.now(),error=repr(e),training_started=False))
            raise


def handoff():
    run.verify();run.bind();DIRECTORY.mkdir(exist_ok=True)
    runtime=run.read(run.OUT/'runtime.json');assert runtime['state']=='cluster_screening'
    pid=runtime['coordinator_pid']
    args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    assert str(run.SCRIPT) in args and 'supervise' in args,'Unexpected coordinator; no signal sent'
    # Only replace the old coordinator. Its default SIGTERM handler releases the
    # flock without setting stop; node supervisors/workers remain untouched.
    run.write(DIRECTORY/'before_handoff.json',dict(runtime=runtime,control=run.read(run.SCHED/'control.json'),
        old_coordinator_pid=pid,utc=run.full.now()))
    run.write(DIRECTORY/'contract.json',dict(policy=POLICY,code_sha256=run.sha(SCRIPT),
        quality_contract_sha256=run.sha(run.OUT/'contract.json'),utc=run.full.now(),
        authorization='用户要求最大限度提速；只调整调度，保持切分/音质/无视觉筛选契约。'))
    verify()
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    while run.cluster.alive(pid):
        if time.monotonic()>deadline:raise RuntimeError('Old coordinator did not exit')
        time.sleep(.1)
    with (DIRECTORY/'coordinator.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'supervise'],stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,start_new_session=True)
    run.write(DIRECTORY/'launch.json',dict(utc=run.full.now(),pid=process.pid,old_pid=pid,workers_restarted=False))
    print(json.dumps(run.read(DIRECTORY/'launch.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['handoff','supervise','verify']);a=p.parse_args()
    if a.stage=='handoff':handoff()
    elif a.stage=='supervise':supervise()
    else:verify();print('Quality and governor contracts verified')
