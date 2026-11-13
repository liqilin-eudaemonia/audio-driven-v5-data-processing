#!/usr/bin/env python3
"""Node17 polling, benchmark selection and persistent annotation supervision."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,acquire

PYTHON='/private/software/conda/envs/vllm/bin/python'


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except FileNotFoundError:return None


def alive(record):return record and identity(record['pid'])==record.get('start_ticks')


def warm_ready(heartbeat,child,selected,revision):
    """A retired worker's last heartbeat must not unlock the next rollout."""
    return bool(child and alive(child) and heartbeat.get('pid')==child.get('pid')
        and heartbeat.get('revision',0)>=revision
        and all(heartbeat.get('execution',{}).get(k)==selected.get(k)
                for k in ('batch_size','tokens','frames','memory'))
        and heartbeat.get('completed_clips',0)>0)


def devices():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    return [dict(zip(('index','memory_used_mib','memory_total_mib','utilization'),map(float,r)))
            for r in csv.reader(io.StringIO(out))]


def gpu_processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    return [int(x) for x in raw.splitlines() if x.strip().isdigit()]


def spawn(work,name,cmd,env=None):
    log=work/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    return {'pid':p.pid,'start_ticks':identity(p.pid),'started_at':time.time(),'cmd':cmd,'log':str(log)}


def choose(work,benchmark):
    summary=read(benchmark/'summary.json');eligible={}
    for name,s in summary.items():
        if s['exit_code'] or s['samples']!=50 or s['truncated_requests']:continue
        # Do not select a faster configuration with worse structural validity.
        full_ok=0
        for path in (benchmark/name/'results').glob('*.json'):
            r=read(path);full=next(v for v in r['records'] if v['view']=='full')
            full_ok+=full['status']=='valid'
        if full_ok==50:
            j=s['settings'];key=tuple(j[k] for k in ('batch','tokens','frames','clips','memory'))
            eligible.setdefault(key,[]).append((name,s))
    if not eligible:raise RuntimeError('No complete, structurally usable benchmark arm')
    # Aggregate repeat durations instead of selecting a lucky fastest card.
    def ranking(items):
        rate=len(items)/sum(1/s['clips_per_s'] for _,s in items)
        return max(s['invalid_requests'] for _,s in items),-rate
    repeats=min(eligible.values(),key=ranking)
    invalid,negative_rate=ranking(repeats);name,best=repeats[0]
    j=best['settings']
    selected={'batch_size':j['batch'],'tokens':j['tokens'],'frames':j['frames'],'clip_batch_size':j['clips'],
        'memory':j['memory'],'prefetch':1,'decode_workers':4,'decode_slots':32,
        'selected_arm':str(benchmark/name),'measured_clips_per_s':-negative_rate,
        'selection_rule':'minimum invalid count, then aggregate repeat throughput',
        'repeat_arms':[n for n,_ in repeats],
        'benchmark_invalid_requests':invalid,'eager':True}
    write(work/'selected.json',selected);return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--hours',type=float,default=8)
    args=p.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    if socket.gethostname()!='dedicated-developjob-lq-dev-machine-2-taaft':raise SystemExit('Node17 only')
    owner=acquire(work/'supervisor.lock',False)
    if owner is None:raise SystemExit('Supervisor already live')
    state=read(work/'supervisor.json') if (work/'supervisor.json').exists() else {
        'started_at':time.time(),'coverage_deadline':time.time()+args.hours*3600,
        'phase':'waiting_for_idle','children':{},'events':[],'benchmark_attempt':0}
    state.update(pid=os.getpid(),start_ticks=identity(os.getpid()),host=socket.gethostname(),supervisor_stopped=False)
    if state['phase']=='annotation':
        state.setdefault('annotation_started_at',min((c['started_at'] for n,c in state['children'].items()
            if n.startswith('gpu-')),default=time.time()))
        if not (work/'resource_tuning.json').exists():
            selected=read(work/'selected.json')
            write(work/'resource_tuning.json',{k:selected[k] for k in ('decode_workers','decode_slots')})
    env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    base=[PYTHON,str(ROOT/'scripts/listen_j_annotation.py')]
    last_telemetry=0;idle_confirmations=0;low_util=0;stalls={}
    while not stop:
        now=time.time();state['heartbeat']=now
        try:
            for role in ['ingest','prime']:
                script=ROOT/'scripts/listen_j_annotation.py'
                done=(work/f'{role}_complete.json').exists()
                child=state['children'].get(role)
                if script.exists() and not done and not alive(child):
                    state['children'][role]=spawn(work,role,base+[role,'--work',str(work),'--upstream',str(args.upstream)],env)
            gpus=devices();state['gpus']=gpus
            if state['phase']=='waiting_for_idle':
                empty=len(gpus)==8 and all(g['memory_used_mib']<1500 for g in gpus) and not gpu_processes()
                idle_confirmations=idle_confirmations+1 if empty else 0
                state['idle_confirmations']=idle_confirmations
                if idle_confirmations>=2:
                    state['benchmark_attempt']+=1
                    b=work/'benchmarks'/f"attempt-{state['benchmark_attempt']:02d}"
                    state['benchmark_dir']=str(b)
                    cmd=[PYTHON,str(ROOT/'scripts/benchmark_listen_visual_native8.py'),
                         '--manifest',str(work/'benchmark_samples.json'),'--matrix',str(work/'benchmark_matrix.json'),
                         '--output',str(b),'--prepared-media-cache','/dev/shm/listen_j_annotation_benchmark',
                         '--model-loaders','8','--timeout-minutes','15']
                    state['children']['benchmark']=spawn(work,'benchmark',cmd,env)
                    state['phase']='benchmark';state['events'].append({'at':now,'event':'eight_idle_gpus_benchmark_started'})
            elif state['phase']=='benchmark':
                b=Path(state['benchmark_dir'])
                if not alive(state['children'].get('benchmark')):
                    if (b/'summary.json').exists():
                        try:choose(work,b)
                        except RuntimeError:
                            write(work/'selected.json',read(work/'fallback_selected.json'))
                            state['events'].append({'at':now,'event':'no_usable_benchmark_arm_using_verified_batch256'})
                        state['phase']='annotation';state['annotation_started_at']=now
                        selected=read(work/'selected.json')
                        write(work/'resource_tuning.json',{'decode_workers':selected['decode_workers'],'decode_slots':selected['decode_slots']})
                        state['events'].append({'at':now,'event':'selected_and_starting_annotation'})
                    elif state['benchmark_attempt']<2:
                        state['phase']='waiting_for_idle';idle_confirmations=0
                    else:
                        # Completed earlier 8-card evidence remains a usable fallback.
                        fallback=read(work/'fallback_selected.json');write(work/'selected.json',fallback)
                        state['phase']='annotation';state['annotation_started_at']=now
                        write(work/'resource_tuning.json',{k:fallback[k] for k in ('decode_workers','decode_slots')})
                        state['events'].append({'at':now,'event':'benchmark_failed_using_verified_batch256'})
            elif state['phase']=='annotation':
                selected=read(work/'selected.json')
                desired_revision=read(work/'config.json').get('gpu_worker_revision',0)
                warming_gpus=[]
                for gpu in range(8):
                    name=f'gpu-{gpu}';child=state['children'].get(name)
                    f=work/'workers'/f'{name}.json'
                    h=read(f) if f.exists() else {}
                    if child and alive(child) and (h.get('pid')!=child['pid'] or
                            (h.get('revision',0)>=desired_revision and not h.get('completed_clips',0))):
                        warming_gpus.append(name)
                state['warming_gpus']=warming_gpus
                restart=state.get('rolling_restart')
                if restart and not alive(restart['child']):
                    state['events'].append({'at':now,'event':'rolling_restart_exited','worker':restart['name']})
                    state['awaiting_warm_worker']=restart['name']
                    state.pop('rolling_restart',None);restart=None
                warming=state.get('awaiting_warm_worker')
                if warming:
                    f=work/'workers'/f'{warming}.json'
                    h=read(f) if f.exists() else {}
                    current=state['children'].get(warming,{})
                    if warm_ready(h,current,selected,read(work/'config.json').get('gpu_worker_revision',0)):
                        state.pop('awaiting_warm_worker',None);warming=None
                for gpu in range(8):
                    name=f'gpu-{gpu}';child=state['children'].get(name)
                    if not alive(child):
                        state['children'][name]=spawn(work,name,base+['worker','--work',str(work),
                            '--upstream',str(args.upstream),'--gpu',str(gpu)],dict(env,CUDA_VISIBLE_DEVICES=str(gpu)))
                        state['events'].append({'at':now,'event':'worker_started','gpu':gpu})
                        continue
                    record=work/'workers'/f'{name}.json'
                    if not record.exists():continue
                    heartbeat=read(record)
                    if heartbeat.get('pid')!=child['pid']:continue
                    progress=heartbeat.get('last_progress_at',heartbeat.get('started_at',now))
                    device=next((g for g in gpus if int(g['index'])==gpu),{})
                    idle=device.get('utilization',100)<5 and now-progress>600
                    stalls[name]=stalls.get(name,0)+1 if idle else 0
                    # Only terminate our recorded process group. A busy long batch
                    # does not qualify as stalled; checkpoints survive recovery.
                    outdated=any(heartbeat.get('execution',{}).get(k)!=selected.get(k)
                                 for k in ('batch_size','tokens','frames','memory'))
                    outdated=outdated or heartbeat.get('revision',0)<read(work/'config.json').get('gpu_worker_revision',0)
                    if not restart and (stalls[name]>=20 or (outdated and not warming and not warming_gpus)):
                        os.kill(child['pid'],signal.SIGTERM)
                        restart={'name':name,'child':child,'requested_at':now,
                                 'reason':'execution_update' if outdated else 'idle_stall'}
                        state['rolling_restart']=restart
                        state['events'].append({'at':now,'event':'rolling_restart_requested',**restart})
                if restart and alive(restart['child']) and now-restart['requested_at']>600:
                    # Graceful group completion normally exits in a few minutes.
                    # For a stuck own worker, kill only its own session.
                    if os.getpgid(restart['child']['pid'])==restart['child']['pid']:
                        os.killpg(restart['child']['pid'],signal.SIGKILL)
                        state['events'].append({'at':now,'event':'own_worker_stuck_killed','worker':restart['name']})
                # CPU nod supplementation consumes independent queued jobs.
                nod_python=read(work/'config.json').get('nod_python')
                if nod_python and Path(nod_python).exists():
                    for slot in range(read(work/'config.json').get('nod_workers',8)):
                        name=f'nod-{slot}'
                        if not alive(state['children'].get(name)):
                            state['children'][name]=spawn(work,name,[nod_python,str(ROOT/'scripts/listen_j_annotation_nod.py'),
                                '--work',str(work),'--slot',str(slot)],dict(env,CUDA_VISIBLE_DEVICES=''))
                progress={};worker_total=0
                for gpu in range(8):
                    f=work/'workers'/f'gpu-{gpu}.json'
                    if f.exists():worker_total+=read(f).get('completed_clips',0)
                for branch in ['vlm','nod']:
                    q=work/'queues'/branch/'state.json'
                    if q.exists():
                        data=read(q);progress[branch]={'completed_jobs':data['completed'],
                            'completed_clips':data.get('completed_clips',0),'inflight_jobs':len(data['inflight']),
                            'failed_jobs':len(data['failed'])}
                state['progress']=progress
                state['current_worker_completed_clips']=worker_total
                mean_util=sum(g['utilization'] for g in gpus)/len(gpus)
                preparing=0
                for gpu in range(8):
                    record=work/'workers'/f'gpu-{gpu}.json'
                    if record.exists():
                        stage=read(record).get('stage','')
                        preparing+=stage.startswith('preparing')
                low_util=low_util+1 if mean_util<65 and preparing>=3 else 0
                if low_util>=4 and now-state.get('annotation_started_at',now)>180:
                    tuning=read(work/'resource_tuning.json')
                    if tuning['decode_workers']<8 and os.getloadavg()[0]<.85*(os.cpu_count() or 192):
                        tuning.update(decode_workers=8,decode_slots=64,changed_at=now)
                        write(work/'resource_tuning.json',tuning)
                        state['events'].append({'at':now,'event':'increase_bounded_decode_feed','gpu_utilization':mean_util,**tuning})
                    low_util=0
            state['events']=state['events'][-100:]
            state['coverage_elapsed_s']=now-state['started_at']
            state['eight_hour_window_elapsed']=now>=state['coverage_deadline']
            state['error']=None
            if now-last_telemetry>=30:
                with (work/'telemetry.jsonl').open('a') as f:f.write(json.dumps({'at':now,'phase':state['phase'],'gpus':gpus,
                    'progress':state.get('progress',{}),'loadavg':os.getloadavg()})+'\n')
                last_telemetry=now
        except Exception as e:
            state['error']=repr(e);state['events'].append({'at':now,'event':'supervisor_check_error','error':repr(e)})
        write(work/'supervisor.json',state)
        time.sleep(15)
    state.update(stopped_at=time.time(),supervisor_stopped=True);write(work/'supervisor.json',state)
    owner.close()


if __name__=='__main__':main()
