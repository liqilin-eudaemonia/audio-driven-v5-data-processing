"""Cross-host scheduling adapter for the frozen, reviewed V8 QC4 pipeline."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
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
import traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_qc4_full as pipeline
from p3_listen.common import read,write,sha
full=pipeline.full
ROOT=pipeline.ROOT
OUT=pipeline.OUT
SCHED=OUT/'cluster'
CONFIG=ROOT/'configs/p3_listen_v8_cluster.json'
SCRIPT=Path(__file__).resolve()
EXP=ROOT/'experiments/p3_listen_v8_cluster_20261023'

def configuration():return read(CONFIG)
def node_config(name):return next(n for n in configuration()['nodes'] if n['name']==name)
def verify():
    pipeline.install();full.verify()
    c=read(SCHED/'contract.json')
    assert c['adapter_sha256']==sha(SCRIPT) and c['config_sha256']==sha(CONFIG)
    assert c['quality_contract_sha256']==sha(OUT/'contract.json')
    assert c['queue_sha256']==sha(SCHED/'queue.json')
    return c

def ssh(node,command):
    c=configuration();args=['ssh','-i',c['ssh_key'],'-p',str(node['port']),'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
        'root@'+node['host'],shlex.join(command)]
    for attempt in range(5):
        p=subprocess.run(args,capture_output=True,text=True,timeout=40)
        if p.returncode==0:return p.stdout
        if p.returncode!=255 or 'kex_exchange_identification' not in p.stderr:break
        time.sleep(1)
    raise RuntimeError(node['name']+': '+p.stderr[-2000:])

def dimensions(models,prep):
    c=configuration()
    if models not in (1,2,4) or prep<1 or prep>8 or 8*models*prep>c['maximum_preparation_slots_per_node']:
        raise ValueError('Unsafe/unvalidated scheduler dimensions')
    return dict(models_per_gpu=models,prep_per_worker=prep,model_workers=8*models,preparation_slots=8*models*prep)

def claim(queue,rank):
    with (SCHED/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(SCHED/'queue_state.json');i=state['cursor']
        if i>=len(queue) or read(SCHED/'control.json').get('stop'):return None
        pair=queue[i]
        # A previous completed result is never overwritten on resume.
        while (OUT/'results'/(pair+'.json')).exists():
            i+=1
            if i>=len(queue):
                write(SCHED/'queue_state.json',dict(cursor=i,total=len(queue)));return None
            pair=queue[i]
        write(SCHED/'claims'/(pair+'.json'),dict(pair=pair,rank=rank,host=socket.gethostname(),pid=os.getpid(),utc=full.now()))
        write(SCHED/'queue_state.json',dict(cursor=i+1,total=len(queue)))
        return pair

def exhausted():
    s=read(SCHED/'queue_state.json');return s['cursor']>=s['total']

def worker(node_name,local_rank):
    c=verify();node=node_config(node_name);rank=node['index']*32+local_rank
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    full.configure_normalization()
    lookup={r['pair']:r for r in read(OUT/'catalog.json')['rows']};queue=read(SCHED/'queue.json')['pairs']
    rejects=read(OUT/'human_rejections.json')['rows'];positives=read(OUT/'human_retained.json')['rows']
    counts=Counter();timing=Counter();started=time.monotonic();failures=0
    def report(stage,pending=0,**kw):
        write(SCHED/'progress'/f'rank_{rank:03d}.json',dict(utc=full.now(),unix=time.time(),host=socket.gethostname(),node=node_name,rank=rank,local_rank=local_rank,gpu=local_rank%8,pid=os.getpid(),
            stage=stage,pending_preparation=pending,counts=dict(counts),timing_seconds=dict(timing),wall_seconds=time.monotonic()-started,**kw))
    report('loading_models')
    contract=read(OUT/'contract.json')
    ast=full.legacy.EventModel(full.legacy.MODEL,batch_size=contract['ast_batch'])
    panns=full.legacy.PannsModel(batch_size=contract['panns_batch'])
    # Threads are created lazily; fill() enforces the current per-node budget.
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending={}
        def fill():
            control=read(SCHED/'control.json');setting=control['nodes'][node_name]
            active=local_rank<setting['model_workers'] and not control.get('stop') and not control.get('drain')
            target=setting['prep_per_worker'] if active else 0
            while len(pending)<target:
                pair=claim(queue,rank)
                if pair is None:break
                pending[pool.submit(full.prepare,lookup[pair])]=pair;counts['claimed_pairs']+=1
            return active
        while True:
            active=fill()
            if not pending:
                if exhausted() or read(SCHED/'control.json').get('stop'):break
                report('idle_disabled' if not active else 'waiting');time.sleep(2);continue
            before=time.monotonic();done,_=wait(pending,return_when=FIRST_COMPLETED)
            timing['waiting_for_preparation']+=time.monotonic()-before
            for future in done:
                pair=pending.pop(future);item=lookup[pair];fill();report('screening',len(pending),pair=pair)
                try:
                    audit,norm,elapsed=future.result();timing['preparation_sum']+=elapsed
                    before=time.monotonic();clips=[full.screen(r,norm,ast,panns,rejects,positives) for r in audit['clips']]
                    timing['audio_screening']+=time.monotonic()-before
                    write(OUT/'results'/(pair+'.json'),dict(pair=pair,split=item['split'],group_id=item['group_id'],complete=True,utc=full.now(),clips=clips,
                        rejected=audit['unresolved'],source_ok=audit['ok'],source_seconds=audit.get('source_seconds'),training_ready=False,
                        scheduler=dict(revision=c['revision'],node=node_name,rank=rank)))
                    counts['complete_pairs']+=1;counts['clips']+=len(clips);failures=0
                except Exception as e:
                    failures+=1;counts['error_pairs']+=1
                    write(OUT/'errors'/(pair+'.json'),dict(pair=pair,utc=full.now(),error=repr(e),traceback=traceback.format_exc(),node=node_name,rank=rank))
                    if failures>=10:raise RuntimeError('Ten consecutive processing errors') from e
                report('pair_finished',len(pending),pair=pair)
    report('finished')

def sample_node(previous):
    cur=full.cpu_sample();delta=max(1,cur[0]-previous[0])
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    return cur,dict(cpu_busy_percent=100*(1-(cur[1]-previous[1])/delta),iowait_percent=100*(cur[2]-previous[2])/delta,
        memory=mem,gpus=[dict(zip(('gpu','utilization_percent','memory_mib','total_mib'),map(int,line.split(',')))) for line in gpu.stdout.strip().splitlines()])

def launch_node(name):
    verify();node=node_config(name)
    with (SCHED/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children={};previous=full.cpu_sample()
        try:
            while True:
                control=read(SCHED/'control.json');setting=control['nodes'][name]
                if not control.get('stop') and not control.get('drain') and not exhausted():
                    for local_rank in range(setting['model_workers']):
                        if local_rank in children:continue
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(local_rank%8),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                        with (SCHED/'logs'/f'{name}_worker_{local_rank:02d}.log').open('ab',buffering=0) as log:
                            children[local_rank]=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'worker','--node',name,'--local-rank',str(local_rank)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                codes={str(i):p.poll() for i,p in children.items()};failure=any(v not in (None,0) for v in codes.values())
                previous,metrics=sample_node(previous)
                finished=(exhausted() or control.get('stop') or failure) and all(v is not None for v in codes.values())
                value=dict(utc=full.now(),unix=time.time(),node=name,host=socket.gethostname(),supervisor_pid=os.getpid(),state='failed' if failure else 'finished' if finished else 'running',
                    children={str(i):dict(pid=p.pid,exit_code=p.poll()) for i,p in children.items()},setting=setting,**metrics)
                write(SCHED/'nodes'/(name+'.json'),value)
                with (SCHED/'telemetry'/(name+'.jsonl')).open('a') as f:f.write(json.dumps(value)+'\n')
                if failure:raise RuntimeError('A node worker failed; see worker logs')
                if finished:break
                time.sleep(10)
        finally:
            # On node supervisor failure, stop only its own children; uncommitted
            # claims are retried by an explicit resume after all nodes have stopped.
            for p in children.values():
                if p.poll() is None:p.terminate()
            for p in children.values():
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()

def detach_node(name):
    verify()
    with (SCHED/'logs'/(name+'_supervisor.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'launch-node','--node',name],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(json.dumps(dict(node=name,pid=p.pid)),flush=True)

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False

def drain_original():
    runtime=read(OUT/'runtime.json');pid=runtime['supervisor_pid'];pids=runtime['pids']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(pipeline.SCRIPT) in cmd and ' run' in cmd,'Not the expected local supervisor'
    write(EXP/'before_runtime.json',runtime)
    os.kill(pid,signal.SIGSTOP)
    saved=None
    try:
        with (OUT/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);saved=read(OUT/'queue_state.json')
            write(EXP/'before_queue_state.json',saved)
            write(OUT/'queue_state.json',dict(cursor=saved['total'],total=saved['total'],reason='Drain for reviewed cross-host scheduling migration'))
        deadline=time.monotonic()+300
        while any(alive(p) for p in pids):
            if time.monotonic()>deadline:raise TimeoutError('Original workers did not drain within 300 seconds')
            time.sleep(2)
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(30):
            if not alive(pid):break
            time.sleep(1)
        assert not alive(pid),'Original supervisor still active'
        write(EXP/'drain_result.json',dict(utc=full.now(),supervisor=pid,workers=pids,all_workers_drained=True,old_supervisor_stopped=True))
    except BaseException:
        if alive(pid):
            if saved is not None:write(OUT/'queue_state.json',saved)
            os.kill(pid,signal.SIGCONT)
        raise

def initialize():
    pipeline.install();full.verify()
    if (SCHED/'contract.json').exists():raise RuntimeError('Cluster already initialized; do not reset a live queue')
    EXP.mkdir(parents=True,exist_ok=True)
    for d in ('claims','logs','nodes','progress','telemetry'):(SCHED/d).mkdir(parents=True,exist_ok=True)
    drain_original()
    # Snapshot completed hashes at cutover: retained outputs must stay unchanged.
    completed={p.stem:sha(p) for p in (OUT/'results').glob('*.json')}
    write(EXP/'preserved_result_hashes.json',completed)
    rows=read(OUT/'catalog.json')['rows'];queue=[r['pair'] for r in rows if r['pair'] not in completed]
    import random
    random.Random(2026102304).shuffle(queue)
    write(SCHED/'queue.json',dict(pairs=queue));write(SCHED/'queue_state.json',dict(cursor=0,total=len(queue)))
    write(SCHED/'control.json',dict(stop=False,stage='benchmark_1x1',nodes={n['name']:dimensions(1,1) for n in configuration()['nodes']}))
    contract=dict(revision=configuration()['revision'],utc=full.now(),adapter_sha256=sha(SCRIPT),config_sha256=sha(CONFIG),quality_contract_sha256=sha(OUT/'contract.json'),
        queue_sha256=sha(SCHED/'queue.json'),previous_complete_pairs=len(completed),pending_pairs=len(queue),science_unchanged=True)
    write(SCHED/'contract.json',contract);verify()
    write(OUT/'runtime.json',dict(state='cluster_ready',utc=full.now(),workers=0,cluster=str(SCHED),training_started=False))

def progress():
    states=[read(p) for p in (SCHED/'progress').glob('*.json')]
    counts=Counter()
    for s in states:counts.update(s['counts'])
    nodes={p.stem:read(p) for p in (SCHED/'nodes').glob('*.json')}
    return dict(unix=time.time(),utc=full.now(),counts=dict(counts),nodes=nodes,queue=read(SCHED/'queue_state.json'))

def pressure(node):
    c=configuration();reasons=[]
    if node['cpu_busy_percent']>c['cpu_busy_backoff_percent']:reasons.append('cpu')
    if node['memory']['MemAvailable']/node['memory']['MemTotal']<c['minimum_available_memory_fraction']:reasons.append('ram')
    if any(g['total_mib']-g['memory_mib']<c['minimum_free_gpu_memory_mib'] for g in node['gpus']):reasons.append('gpu_memory')
    return reasons

def set_stage(models,prep,label):
    # Drain before adding workers: old prefetched tasks must not overlap a larger
    # new worker population and exceed the per-node preparation budget.
    ctrl=read(SCHED/'control.json');ctrl.update(stage='draining_for_'+label,drain=True);write(SCHED/'control.json',ctrl)
    deadline=time.monotonic()+300
    while True:
        states=[read(p) for p in (SCHED/'progress').glob('*.json')]
        if states and all(s['stage'] in ('idle_disabled','finished') for s in states):break
        if time.monotonic()>deadline:raise TimeoutError('Workers did not drain for concurrency change')
        time.sleep(2)
    ctrl.update(stage=label,drain=False,nodes={n['name']:dimensions(models,prep) for n in configuration()['nodes']});write(SCHED/'control.json',ctrl)

def supervise():
    verify();cfg=configuration()
    with (SCHED/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for n in cfg['nodes']:
                print(ssh(n,[cfg['python'],str(SCRIPT),'detach-node','--node',n['name']]),flush=True)
        except BaseException:
            ctrl=read(SCHED/'control.json');ctrl['stop']=True;write(SCHED/'control.json',ctrl)
            raise
        stages=cfg['stages'];stage_index=0;stage_start=time.time();baseline=None;trials=[];best=None;last_report=0;began=time.time();backoff=set()
        try:
            while True:
                snap=progress();nodes=snap['nodes'];now=time.time()
                if len(nodes)==3 and all(v['state']=='finished' for v in nodes.values()):break
                for name,node in nodes.items():
                    if node['state']=='failed':raise RuntimeError(name+' worker failure')
                    if now-node['unix']>90:raise RuntimeError(name+' heartbeat older than 90 seconds')
                    p=pressure(node)
                    if p:
                        ctrl=read(SCHED/'control.json');d=ctrl['nodes'][name]
                        ctrl['nodes'][name]=dimensions(1,max(1,d['prep_per_worker']//2));ctrl['stage']='resource_backoff'
                        write(SCHED/'control.json',ctrl);backoff.add(name)
                        with (SCHED/'safety_events.jsonl').open('a') as f:f.write(json.dumps(dict(utc=full.now(),node=name,reasons=p,setting=ctrl['nodes'][name]))+'\n')
                if now-began>90 and len(nodes)!=3:raise RuntimeError('Missing node heartbeat')
                elapsed=now-stage_start
                if baseline is None and elapsed>=cfg['warmup_seconds'] and len(nodes)==3:baseline=snap
                if baseline and elapsed>=cfg['benchmark_seconds'] and stage_index<len(stages):
                    secs=snap['unix']-baseline['unix'];pairs=snap['counts'].get('complete_pairs',0)-baseline['counts'].get('complete_pairs',0)
                    trial=dict(stage=stages[stage_index],seconds=secs,completed_pairs=pairs,pairs_per_minute=pairs*60/max(secs,1),
                        errors=snap['counts'].get('error_pairs',0)-baseline['counts'].get('error_pairs',0),resource_backoff_nodes=sorted(backoff),utc=full.now())
                    trials.append(trial)
                    if not backoff and (best is None or trial['pairs_per_minute']>best['pairs_per_minute']*(1+cfg['minimum_throughput_gain'])):best=trial
                    stage_index+=1
                    # Resource pressure ends the sweep; preserve the reduced per-node setting.
                    if backoff:stage_index=len(stages)
                    if stage_index<len(stages):
                        m,p=stages[stage_index];set_stage(m,p,f'benchmark_{m}x{p}');stage_start=time.time();baseline=None
                    elif not backoff and best:
                        m,p=best['stage'];set_stage(m,p,'selected_by_observed_throughput')
                    write(SCHED/'benchmark.json',dict(trials=trials,selected=best,resource_backoff_nodes=sorted(backoff),
                        note='Live randomized disjoint production workload; observed rates are not matched-input speedup or a global optimum. Models and batches unchanged.'))
                ctrl=read(SCHED/'control.json')
                runtime=dict(state='cluster_screening',utc=full.now(),coordinator_pid=os.getpid(),cluster=str(SCHED),nodes=list(nodes),control=ctrl,
                    workers=sum(d['model_workers'] for d in ctrl['nodes'].values()),total_preparation_slots=sum(d['preparation_slots'] for d in ctrl['nodes'].values()),
                    complete_pairs=read(SCHED/'contract.json')['previous_complete_pairs']+snap['counts'].get('complete_pairs',0),error_pairs=snap['counts'].get('error_pairs',0),training_started=False,pipeline_render_after_screening=True)
                write(OUT/'runtime.json',runtime);write(SCHED/'latest.json',snap)
                if now-last_report>120:pipeline.report();last_report=now
                time.sleep(10)
            # All supervisors acknowledge completion before report/export/render.
            summary=pipeline.report(export=True)
            assert not (set(read(SCHED/'queue.json')['pairs'])-{p.stem for p in (OUT/'results').glob('*.json')}-{p.stem for p in (OUT/'errors').glob('*.json')}),'Unaccounted claims'
            write(OUT/'runtime.json',dict(state='rendering_candidates',utc=full.now(),workers=4,cluster=str(SCHED),training_started=False))
            pipeline.render()
            write(OUT/'runtime.json',dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',utc=full.now(),workers=0,summary=summary,cluster=str(SCHED),training_started=False))
        except BaseException as e:
            ctrl=read(SCHED/'control.json');ctrl['stop']=True;write(SCHED/'control.json',ctrl)
            write(OUT/'runtime.json',dict(state='cluster_stopping_after_error',utc=full.now(),error=repr(e),cluster=str(SCHED),training_started=False))
            raise

def detach():
    verify()
    with (SCHED/'logs/coordinator.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'supervise'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    value=dict(utc=full.now(),pid=p.pid);write(SCHED/'launch.json',value);print(json.dumps(value),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','worker','launch-node','detach-node','supervise','detach','verify'])
    p.add_argument('--node',default='node12');p.add_argument('--local-rank',type=int,default=0);a=p.parse_args()
    if a.stage=='initialize':initialize()
    elif a.stage=='worker':worker(a.node,a.local_rank)
    elif a.stage=='launch-node':launch_node(a.node)
    elif a.stage=='detach-node':detach_node(a.node)
    elif a.stage=='supervise':supervise()
    elif a.stage=='detach':detach()
    else:verify();print('Frozen quality and cluster contracts verified')
