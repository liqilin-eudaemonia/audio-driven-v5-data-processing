"""Durable annotation -> condition caches -> saved checkpoint -> AV metrics queue."""
import os, sys, time, signal, subprocess, fcntl, uuid
from functools import lru_cache
from pathlib import Path
from eval_common import HERE,ROOT,PY,TRAIN,read,write,sha,prepare,environment,ready
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import digest
from p2_glare_eval.common import artifact_valid

def training_allows():
    state=read(TRAIN/'training_queue/state.json')
    running=[j for j in state['jobs'].values() if j['status']=='running']
    if running:return all(j['stage']==1 and j['step']<1100 for j in running)
    # Keep all devices vacant during stage reload and automatic LoRA benchmarks.
    accel=ROOT/'experiments/p2_full50h_omni_accel_20261028/status.json'
    return state['status'] in ('complete','completed') and accel.exists() and read(accel).get('phase') in ('complete','completed')

def terminate(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def valid_generation(job):
    out=Path(job['output']);p=out/'complete.json'
    if not p.exists():return False
    m=read(p)
    files=tuple((n,(out/n).stat().st_size,(out/n).stat().st_mtime_ns) for n in m['files'])
    return _valid_generation(str(out),sha(p),files,job['sample_id'],job['seed'],read(Path(job['checkpoint']).with_suffix('.complete.json'))['sha256'])

@lru_cache(maxsize=1024)
def _valid_generation(path,marker_sha,files,sid,seed,checkpoint_sha):
    out=Path(path);m=read(out/'complete.json')
    if not artifact_valid(out,m['contract']):return False
    meta=read(out/'completed.json')
    return meta['sample_id']==sid and meta['sample_seed']==seed and meta['checkpoint_sha256']==checkpoint_sha

def tasks(jobs):
    ids=read(HERE/'plans/protocol.json')['sample_ids'];result=[]
    for i in range(0,len(ids),6):
        p=HERE/'tasks'/f'ids{i:03d}.json';write(p,ids[i:i+6])
        for part in ['text','input_plan']:
            result.append(dict(id=f'cache_{part}_{i}',kind='cache',cmd=[str(PY),str(HERE/'eval_worker.py'),'cache',part,str(p)]))
    oldrun=next(j['run'] for j in jobs if j['family']=='legacy' and j['arm']=='legacy_E1F')
    p=HERE/'tasks/legacy_ids.json';write(p,[o['sample_id'] for o in read(HERE/'test48/observations.json')])
    result.append(dict(id='legacy_text',kind='cache',cmd=[str(PY),str(HERE/'eval_worker.py'),'legacy-cache',oldrun,str(p)]))
    for i,j in enumerate(jobs):
        p=HERE/'tasks'/f'generate_{i:04d}.json';write(p,j)
        result.append(dict(id=f'generate_{i:04d}',kind='generate',job=j,cmd=[str(PY),str(HERE/'eval_worker.py'),'generate',str(p)]))
    for arm,split in dict.fromkeys((j['arm'],j['split']) for j in jobs):
        result.append(dict(id=f'metrics_{arm}_{split}',kind='metrics',arm=arm,split=split,
            cmd=[str(PY),str(HERE/'eval_metrics.py'),'evaluate',arm,split]))
    return result

def main():
    lock=(HERE/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (HERE/'test48/COMPLETE.json').exists():
        write(HERE/'eval_status.json',dict(phase='waiting_full_test48_labels',at=time.time(),checkpoint_policy='final1249_only'))
        if (HERE/'EVAL_STOP').exists():return
        time.sleep(10)
    while not (HERE/'pilot/COMPLETE.json').exists():
        write(HERE/'eval_status.json',dict(phase='waiting_real_gpu_pilot',at=time.time()))
        if (HERE/'EVAL_STOP').exists():return
        time.sleep(10)
    jobs=prepare();queue=tasks(jobs);(HERE/'logs').mkdir(exist_ok=True)
    code={str(HERE/n):sha(HERE/n) for n in ['eval_common.py','eval_worker.py','eval_metrics.py','eval_chain.py']}
    frozen=dict(code=code,jobs_sha256=sha(HERE/'jobs.json'),plan_sha256=sha(HERE/'plans/protocol.json'))
    if (HERE/'EVAL_FROZEN.json').exists():assert read(HERE/'EVAL_FROZEN.json')==frozen
    else:write(HERE/'EVAL_FROZEN.json',frozen)
    active={};done=set();errors={};deferred_until={};gpu_slots=[4,5,6]
    def completed(t):
        if t['kind']=='generate':return valid_generation(t['job'])
        p=HERE/'task_results'/(t['id']+'.json')
        return p.exists() and read(p).get('status')=='complete' and read(p).get('frozen')==digest(frozen)
    for t in queue:
        if completed(t):done.add(t['id'])
    try:
        while len(done)<len(queue):
            if any(sha(p)!=h for p,h in code.items()):raise RuntimeError('Evaluation source changed while armed')
            allow=training_allows() and not (HERE/'EVAL_STOP').exists()
            for gpu,(p,t,started,logpath) in list(active.items()):
                cancelled=not allow
                timedout=time.time()-started>3600
                if cancelled or timedout:terminate(p)
                if p.poll() is None:continue
                del active[gpu]
                success=p.returncode==0 and (t['kind']!='generate' or valid_generation(t['job']))
                if success:
                    done.add(t['id']);write(HERE/'task_results'/(t['id']+'.json'),dict(status='complete',frozen=digest(frozen),seconds=time.time()-started,log=str(logpath)))
                else:
                    if t['kind']=='generate':
                        dest=Path(t['job']['output'])
                        if dest.exists() and not (dest/'complete.json').exists():
                            archive=HERE/'failed_attempts'/(t['id']+'_'+uuid.uuid4().hex);archive.parent.mkdir(exist_ok=True);dest.rename(archive)
                    if not cancelled:
                        errors[t['id']]=errors.get(t['id'],0)+1;deferred_until[t['id']]=time.time()+60
                    write(HERE/'task_results'/(t['id']+'.json'),dict(status='yielded' if cancelled else 'retry',returncode=p.returncode,attempts=errors.get(t['id'],0),log=str(logpath)))
            cache_done=all(t['id'] in done for t in queue if t['kind']=='cache')
            generation_done=sum(t['kind']=='generate' and t['id'] in done for t in queue)
            used={t['id'] for _,t,_,_ in active.values()}
            if allow:
                free={int(a):int(b) for a,b in [l.split(',') for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()]}
                for gpu in gpu_slots:
                    if gpu in active or free[gpu]<80000:continue
                    # First real generation is a single-GPU memory/correctness pilot.
                    if cache_done and generation_done==0 and active:break
                    candidates=[]
                    for t in queue:
                        if t['id'] in done or t['id'] in used or errors.get(t['id'],0)>=3 or deferred_until.get(t['id'],0)>time.time():continue
                        if t['kind']!='cache' and not cache_done:continue
                        if t['kind']=='generate' and not ready(t['job']['checkpoint']):continue
                        if t['kind']=='metrics' and not all(valid_generation(j) for j in jobs if j['arm']==t['arm'] and j['split']==t['split']):continue
                        candidates.append(t)
                    if not candidates:continue
                    candidates.sort(key=lambda t:0 if t['kind']=='metrics' else 1)
                    t=candidates[0];logpath=HERE/'logs'/(t['id']+'_'+uuid.uuid4().hex[:8]+'.log')
                    with logpath.open('w') as log:p=subprocess.Popen(t['cmd'],env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
                    active[gpu]=(p,t,time.time(),logpath);used.add(t['id'])
            blocked=[k for k,v in errors.items() if v>=3]
            write(HERE/'eval_status.json',dict(phase='running' if active else 'waiting_resources_or_checkpoint',at=time.time(),
                tasks_done=len(done),tasks_total=len(queue),generation_done=generation_done,generation_total=len(jobs),
                active=[dict(gpu=g,task=t['id'],pid=p.pid,log=str(l)) for g,(p,t,_,l) in active.items()],failed_tasks=blocked,
                checkpoint_policy='AV/Event final1249; Feature waits final1249; legacy final1268',training_unchanged=True))
            if blocked and not active:raise RuntimeError('Tasks failed after three attempts: '+str(blocked))
            if (HERE/'EVAL_STOP').exists():return
            time.sleep(5)
        write(HERE/'EVAL_COMPLETE.json',dict(status='complete',generation_outputs=len(jobs),frozen=frozen))
        write(HERE/'eval_status.json',dict(phase='complete',generation_done=len(jobs)))
    finally:
        for p,_,_,_ in active.values():terminate(p)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(HERE/'eval_status.json',dict(phase='failed',error=repr(exc),at=time.time()));raise
