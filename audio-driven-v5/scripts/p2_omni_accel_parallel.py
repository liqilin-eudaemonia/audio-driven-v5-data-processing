"""Stage2-only parallel scheduling on the existing duration-aligned training core."""
from pathlib import Path
import copy,json,os,socket,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_duration_execution import install_backend
from p2_listen.contracts import read,sha,atomic_json
from p2_guided_chain import training_queue as native
from p2_legacy_qwen_queue import latest,resources
from listen_cluster import launch
LOSSES=native.LOSSES

def commands(cluster,slots,config,resume=None,parent=None,port=34500,stop_after=None):
    cc,rows=native.commands(cluster,slots,config,resume,parent,port)
    for row in rows:
        row[row.index(str(ROOT/'scripts/p2_guided_chain_train.py'))]=str(ROOT/'scripts/p2_duration_train.py')
        if stop_after is not None:row+=['--stop-after',str(stop_after)]
    return cc,rows

def initial_jobs(root,policy):
    if policy not in ('serial24','parallel6'):raise ValueError('Unknown execution policy')
    parents=read(root/'parents.json');jobs={}
    for i,loss in enumerate(LOSSES):
        slots=[i] if policy=='parallel6' else [0,1,2,3]
        jobs[loss]=dict(stage=2,status='pending',slots=slots,parent=parents[loss]['checkpoint'],
            config=native.prepare_run(root,loss,2,slots),failures=0,start_step=0)
    return jobs

def expansion(jobs,active):
    used={s for job in jobs.values() if job['status']!='complete' for s in job.get('expanding_to',job['slots'])}
    free=sorted(set(range(4))-used)
    if not free or any('expanding_to' in job for job in jobs.values()):return None
    candidates=sorted((loss for loss in active if len(jobs[loss]['slots'])<4),key=lambda k:jobs[k].get('observed_step',0))
    for loss in candidates:
        n=len(jobs[loss]['slots']);growth=3 if n==1 and len(free)==3 else n if n<=len(free) else 0
        if growth:return loss,sorted(jobs[loss]['slots']+free[:growth])
    return None

def verify(root):
    for path,want in read(root/'FROZEN.json').items():
        if sha(path)!=want:raise ValueError('Frozen acceleration source/config changed: '+path)

def launch_checked(cluster,cmds,logs,outputs,cancel=None,stop_file=None):
    """Watch only this job's progress; the existing SSH guard owns its workers."""
    cancel=cancel or threading.Event();done=threading.Event();began=time.time()
    def watchdog():
        while not done.wait(10):
            if stop_file is not None and Path(stop_file).exists():cancel.set();return
            recent=max([began]+[p.stat().st_mtime for out in outputs for p in [Path(out)/'metrics.jsonl'] if p.exists()])
            if time.time()-recent>1200:cancel.set();return
    watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
    try:launch(cluster,cmds,logs,cancel_event=cancel)
    finally:done.set();watcher.join(timeout=1)

def run(root):
    root=Path(root);install_backend();q=root/'training_queue';q.mkdir(exist_ok=True);cluster=read(root/'cluster.json')
    if not (q/'jobs.json').exists():
        decision=read(root/'benchmark/DECISION.json');policy=decision['policy']
        atomic_json(q/'policy.json',dict(policy=policy,reason='GPU benchmark decision'))
        atomic_json(q/'jobs.json',initial_jobs(root,policy))
    jobs=read(q/'jobs.json');policy=read(q/'policy.json')['policy'];active={}
    draining=(q/'fallback_requested.json').exists() and policy=='parallel6'
    def persist():atomic_json(q/'jobs.json',jobs)
    def status(label='running'):
        values={}
        for loss,j in jobs.items():
            c=read(j['config']);p=Path(c['output'])/'status.json';progress=read(p) if p.exists() else {}
            j['observed_step']=progress.get('step',j.get('start_step',0))
            values[loss]=dict(status=j['status'],stage=2,world=len(j['slots'])*6,slots=j['slots'],step=j['observed_step'],
                budget=c['training']['steps'],output=c['output'],parent=j['parent'],error=j.get('error'))
        atomic_json(q/'state.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),status=label,policy=policy,draining=draining,jobs=values))
    try:
        while not (root/'STOP').exists():
            verify(root)
            for loss,j in jobs.items():
                if loss not in active:continue
                work=active[loss]
                if work['thread'].is_alive():continue
                work['thread'].join();del active[loss];c=read(j['config']);cp,step=latest(Path(c['output']));j['start_step']=step
                if cp and step==c['training']['steps']:
                    j.update(status='complete',checkpoint=str(cp));j.pop('expanding_to',None)
                elif draining:
                    j['status']='pending';j.pop('expanding_to',None)
                elif 'expanding_to' in j:
                    if cp is None:raise RuntimeError('Expansion lacks complete checkpoint')
                    j['slots']=j.pop('expanding_to');j['config']=native.prepare_run(root,loss,2,j['slots'],cp);j['status']='pending'
                else:
                    j['failures']+=1;j['error']=work['errors'] or ['Worker exited before full budget'];j['status']='pending'
                    j['retry_after']=time.time()+min(300,15*j['failures'])
                    if policy=='parallel6' and j['failures']>=3:
                        draining=True
                        atomic_json(q/'fallback_requested.json',dict(at=time.time(),loss=loss,error=j['error']))
                persist()
            if draining:
                for loss in active:(Path(read(jobs[loss]['config'])['output'])/'STOP').write_text('Save before automatic serial fallback.\n')
                if not active:
                    # Every survivor keeps its own complete model/optimizer and ordinal position.
                    for loss,j in jobs.items():
                        if j['status']=='complete':continue
                        cp,step=latest(Path(read(j['config'])['output']));j['slots']=[0,1,2,3]
                        j['config']=native.prepare_run(root,loss,2,j['slots'],cp);j.update(status='pending',start_step=step,failures=0,retry_after=0)
                        (Path(read(j['config'])['output'])/'STOP').unlink(missing_ok=True)
                    # Commit migrated jobs before the policy flag. A crash before
                    # the flag safely re-enters draining; never launch stale 6-rank
                    # jobs under a serial24 policy after a controller restart.
                    persist();policy='serial24'
                    atomic_json(q/'policy.json',dict(policy=policy,reason='Parallel job failed repeatedly; saved survivors before fallback'))
                    draining=False
            if all(j['status']=='complete' for j in jobs.values()):status('complete');return
            if not draining:
                inventory=None
                for loss,j in jobs.items():
                    if j['status']=='complete' or loss in active or time.time()<j.get('retry_after',0):continue
                    if policy=='serial24' and (active or loss!=next(k for k in LOSSES if jobs[k]['status']!='complete')):continue
                    if policy=='serial24' and j['slots']!=[0,1,2,3]:raise ValueError('Serial fallback must own all 24 ranks')
                    if 'expanding_to' in j:
                        cp,_=latest(Path(read(j['config'])['output']))
                        if cp is None:raise RuntimeError('Pending migration lacks checkpoint')
                        j['slots']=j.pop('expanding_to');j['config']=native.prepare_run(root,loss,2,j['slots'],cp);persist()
                    c=read(j['config']);template=read(root/'configs'/f'guided_{loss}_s2.json')
                    if {k:v for k,v in c.items() if k!='output'}!={k:v for k,v in template.items() if k!='output'}:raise ValueError('Experiment changed during scheduling')
                    out=Path(c['output']);cp,step=latest(out)
                    if cp and step==c['training']['steps']:j.update(status='complete',checkpoint=str(cp));persist();continue
                    if inventory is None:inventory=resources(cluster)
                    selected={g for s in j['slots'] for g in (2*s,2*s+1)}
                    if any(g[1]<cluster['minimum_free_gib']*1024 for n in inventory for g in n['gpus'] if g[0] in selected):continue
                    if cp is None and (out/'run.json').exists():(out/'run.json').rename(out/f'run.failed_start.{time.time_ns()}.json')
                    j['attempt']=j.get('attempt',0)+1;j.update(status='running',error=None)
                    cc,cmds=commands(cluster,j['slots'],j['config'],cp,j['parent'],34600+LOSSES.index(loss)*100+j['attempt']%90)
                    work=dict(cancel=threading.Event(),errors=[]);logs=q/'attempts'/f'{loss}-{time.time_ns()}'
                    def execute(cc=cc,cmds=cmds,logs=logs,work=work,out=out):
                        try:launch_checked(cc,cmds,logs,[out],work['cancel'],stop_file=root/'STOP')
                        except Exception as exc:work['errors'].append(repr(exc))
                    work['thread']=threading.Thread(target=execute);active[loss]=work;persist();work['thread'].start()
                if policy=='parallel6':
                    request=expansion(jobs,active)
                    if request:
                        loss,slots=request;jobs[loss]['expanding_to']=slots;persist()
                        (Path(read(jobs[loss]['config'])['output'])/'STOP').write_text('Save for GPU expansion.\n')
            status();time.sleep(3)
    finally:
        for work in active.values():work['cancel'].set()
        for work in active.values():work['thread'].join(timeout=120)
