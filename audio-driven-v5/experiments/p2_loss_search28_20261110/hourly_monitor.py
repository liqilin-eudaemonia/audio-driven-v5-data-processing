"""Hourly read-only search monitor; never kills or restarts any process."""
from pathlib import Path
import concurrent.futures
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from listen_cluster import transport
EVAL=ROOT/'experiments/p2_loss_search28_evaluation_20261110'


def process(path):
    if not path.exists():return dict(registered=False,live=False)
    r=read(path);p=Path('/proc')/str(r['pid'])
    if not p.exists():return dict(**r,registered=True,live=False)
    stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return dict(**r,registered=True,live=stat[19]==r['start_ticks']and stat[0]not in('Z','X'),state=stat[0])


def inspect():
    state=read(WORK/'STATE.json')if(WORK/'STATE.json').exists()else{}
    report=dict(at=time.time(),state=state,interval_seconds=3600,readonly=True,
        controllers={k:process(p)for k,p in [('training',WORK/'CONTROLLER.json'),
            ('evaluation',EVAL/'STAGE_CONTROLLER.json'),('search',EVAL/'SEARCH_CONTROLLER.json')]})
    name=state.get('candidate','baseline');out=WORK/'runs'/name;cluster=read(out/'cluster.json')
    if(out/'metrics.jsonl').exists():
        lines=(out/'metrics.jsonl').read_text().splitlines()
        # A concurrently appended final line is not a training failure.
        values=[]
        for line in lines[-60:]:
            try:values.append(json.loads(line))
            except json.JSONDecodeError:continue
        if values:
            report['training']=dict(candidate=name,step=values[-1]['step'],recent_steps=len(values),
                median_step_seconds=statistics.median(r['seconds']for r in values),
                peak_reserved_gib=max(r['peak_reserved_gib']for r in values),
                max_input_wait_ms=max(r['input_wait_seconds']for r in values)*1000,
                all_finite=all(math.isfinite(r['weighted_total'])and all(math.isfinite(v)for v in r['grad_norm'].values())for r in values),
                metric_age_seconds=time.time()-(out/'metrics.jsonl').stat().st_mtime)
    def node(n):
        command=['nvidia-smi','-i',','.join(map(str,n['gpus'])),
            '--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']
        for attempt in range(3):
            try:
                r=subprocess.run(transport(cluster,n,command),capture_output=True,text=True,timeout=35,check=True)
                rows=[[int(x.strip())for x in line.split(',')]for line in r.stdout.splitlines()if line.strip()]
                return dict(node=n['name'],gpus=[dict(index=a,utilization=b,used_mib=c,total_mib=d)for a,b,c,d in rows])
            except(subprocess.CalledProcessError,subprocess.TimeoutExpired)as exc:
                if attempt==2:return dict(node=n['name'],observation_error=str(exc),process_failure_inferred=False)
                time.sleep(3)
    with concurrent.futures.ThreadPoolExecutor(4)as pool:report['resources']=list(pool.map(node,cluster['nodes']))
    report['attention']=[]
    if report.get('training',{}).get('all_finite')is False:report['attention'].append('nonfinite_training_metrics')
    if state.get('stage')=='training' and not(out/f"STAGE_{state['target']}.json").exists():
        if not report['controllers']['training']['live']:report['attention'].append('training_controller_not_live')
        if report.get('training',{}).get('metric_age_seconds',0)>1200:report['attention'].append('training_metrics_older_than20minutes; inspect_actual_workers')
    if not report['controllers']['search']['live']:report['attention'].append('search_controller_not_live')
    folder=WORK/'hourly_monitor';folder.mkdir(exist_ok=True)
    atomic_json(folder/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'),report)
    atomic_json(folder/'LATEST.json',report)
    return report


def main():
    folder=WORK/'hourly_monitor';folder.mkdir(exist_ok=True)
    with(folder/'monitor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        atomic_json(folder/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,at=time.time(),interval_seconds=3600))
        while not(folder/'STOP').exists():
            try:inspect()
            except Exception as exc:atomic_json(folder/'OBSERVATION_ERROR.json',dict(at=time.time(),error=repr(exc),no_processes_restarted=True))
            if'--once'in sys.argv:return
            until=time.monotonic()+3600
            while time.monotonic()<until and not(folder/'STOP').exists():time.sleep(min(30,until-time.monotonic()))


if __name__=='__main__':main()
