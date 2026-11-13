#!/usr/bin/env python3
"""Four concurrent six-rank jobs; keep batch24 and grow survivors to 12/24 ranks."""
from pathlib import Path
import copy,fcntl,json,os,socket,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
from p2_legacy_qwen_queue import latest,resources
from listen_cluster import launch
ARMS=('E0','E1','E0F','E1F')

def commands(cluster,slots,config,resume,port):
    cc=copy.deepcopy(cluster);gpus=sorted(i for slot in slots for i in (2*slot,2*slot+1));result=[]
    if len(slots) not in (1,2,4):raise ValueError('Only world6,12,24 preserve supported accumulation')
    for rank,node in enumerate(cc['nodes']):
        node['gpus']=gpus
        cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'PYTHONUNBUFFERED=1','TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cc['network'].items()],cc['python'],'-m','torch.distributed.run','--max_restarts=0','--nnodes=3','--nproc_per_node='+str(len(gpus)),'--node_rank='+str(rank),'--master_addr='+cc['nodes'][0]['host'],'--master_port='+str(port),str(ROOT/'scripts/p2_legacy_qwen_lora.py'),'train','--config',str(config)]
        if resume:cmd+=['--resume',str(resume)]
        result.append(cmd)
    return cc,result

def migration(source,config,world):
    from p2_matrix16.checkpoint import checked
    from p2_legacy_qwen_lora import backend
    from p2_listen.cache import atomic_torch
    saved=checked(source);identity=copy.deepcopy(saved['identity']);before=copy.deepcopy(identity['experiment']);after=copy.deepcopy(config)
    before.pop('output');after.pop('output')
    if before!=after or world not in (6,12,24) or identity['world'] not in (6,8,12,24):raise ValueError('Migration may only change topology and output path')
    if identity['source']!=backend.source_identity() or identity['data_identity']!=backend.data_identity(config):raise ValueError('Migration source/data changed')
    transition=dict(checkpoint=str(Path(source).resolve()),sha256=sha(source),from_world=identity['world'],to_world=world,step=saved['step'],global_batch=24,
        rng_rank_mapping=[i%len(saved['rank_rng']) for i in range(world)],seeding='Every sample resets CPU/current CUDA seed and decoder generator by original global ordinal; no stochastic data augmentation')
    identity['world']=world;identity['experiment']=config
    identity['parent']['execution_transitions']=identity['parent'].get('execution_transitions',[])+[transition]
    value=dict(saved,identity=identity,rank_rng=[saved['rank_rng'][i] for i in transition['rng_rank_mapping']]);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    path=out/Path(source).name;atomic_torch(path,value);atomic_json(path.with_suffix('.complete.json'),dict(sha256=sha(path),step=saved['step']));atomic_json(out/'run.json',identity,immutable=True);atomic_json(out/'topology_migration.json',dict(**transition,target_sha256=sha(path),model_and_optimizer_tensors_unmodified=True))
    return path

def prepare(root,arm,slots,source=None):
    template=read(root/'configs'/f'{arm}.json');world=6*len(slots);step=read(Path(source).with_suffix('.complete.json'))['step'] if source else 0
    base=root/'parallel_runs'/arm/f'w{world}_from_{step:06d}';c=copy.deepcopy(template);c['output']=str(base)
    config=root/'parallel_configs'/f'{arm}_w{world}_from_{step:06d}.json';atomic_json(config,c,immutable=True)
    atomic_json(base/'preflight.json',read(Path(template['output'])/'preflight.json'))
    if source and not (base/Path(source).with_suffix('.complete.json').name).exists():migration(source,c,world)
    return str(config)

def run(root):
    q=root/'parallel_queue';cluster=read(root/'parallel_cluster.json');frozen=read(q/'frozen.json');active={}
    def verify():
        for path,want in frozen.items():
            if sha(path)!=want:raise ValueError('Frozen parallel source changed: '+path)
    def persist():atomic_json(q/'jobs.json',jobs)
    def status(label='running',**extra):
        values={}
        for arm,job in jobs.items():
            c=read(job['config']);p=Path(c['output'])/'status.json';progress=read(p) if p.exists() else {}
            values[arm]=dict(status=job['status'],world=6*len(job['slots']),slots=job['slots'],step=progress.get('step',job.get('start_step',0)),budget=1268,output=c['output'],**({'error':job['error']} if job.get('error') else {}))
        atomic_json(q/'state.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),status=label,jobs=values,**extra))
    jobs=read(q/'jobs.json')
    try:
        while True:
            if (q/'STOP').exists():status('stopping');return
            verify()
            for arm,job in jobs.items():
                if arm in active:
                    work=active[arm]
                    if work['thread'].is_alive():
                        metric=Path(read(job['config'])['output'])/'metrics.jsonl'
                        if time.time()-max(work['began'],metric.stat().st_mtime if metric.exists() else 0)>1200:work['cancel'].set()
                        continue
                    work['thread'].join();del active[arm]
                    cp,step=latest(Path(read(job['config'])['output']));job['start_step']=step
                    if cp and step==1268:job['status']='complete';job.pop('expanding_to',None)
                    elif 'expanding_to' in job:
                        if cp is None:raise RuntimeError('Expansion lacks a complete checkpoint')
                        job['slots']=job.pop('expanding_to');job['config']=prepare(root,arm,job['slots'],cp);job['status']='pending'
                    else:
                        job['failures']=job.get('failures',0)+1;job['error']=work['result'];job['status']='pending' if job['failures']<12 else 'failed';job['retry_after']=time.time()+15
                    persist()
                if job['status'] in ('complete','failed') or arm in active:continue
                if time.time()<job.get('retry_after',0):continue
                # A controller may have died while a graceful expansion was pending.
                if 'expanding_to' in job:
                    cp,step=latest(Path(read(job['config'])['output']))
                    if cp is None:raise RuntimeError('Cannot recover pending topology transition')
                    job['slots']=job.pop('expanding_to');job['config']=prepare(root,arm,job['slots'],cp);job['start_step']=step;persist()
                c=read(job['config']);template=read(root/'configs'/f'{arm}.json')
                if {k:v for k,v in c.items() if k!='output'}!={k:v for k,v in template.items() if k!='output'}:raise ValueError('Parallel experiment hyperparameters changed')
                cp,step=latest(Path(c['output']))
                if cp and step==1268:job['status']='complete';persist();continue
                try:inventory=resources(cluster)
                except Exception as e:job['error']=str(e);continue
                selected={i for slot in job['slots'] for i in (slot*2,slot*2+1)}
                if any(g[1]<cluster['minimum_free_gib']*1024 for n in inventory for g in n['gpus'] if g[0] in selected):continue
                out=Path(c['output'])
                if cp is None and (out/'run.json').exists():(out/'run.json').rename(out/f'run.failed_start.{time.time_ns()}.json')
                job['attempt']=job.get('attempt',0)+1;job['status']='running';job['error']=None
                port=30010+ARMS.index(arm)*100+job['attempt']%90;cc,cmd=commands(cluster,job['slots'],job['config'],cp,port)
                work=dict(cancel=threading.Event(),result=[],began=time.time());logs=q/'attempts'/f'{arm}-{time.time_ns()}'
                def execute(cc=cc,cmd=cmd,work=work,logs=logs):
                    try:launch(cc,cmd,logs,cancel_event=work['cancel']);work['result'].append('success')
                    except Exception as e:work['result'].append(str(e))
                work['thread']=threading.Thread(target=execute);active[arm]=work;persist();work['thread'].start()
            if all(j['status']=='complete' for j in jobs.values()):status('complete');return
            if not active and all(j['status'] in ('complete','failed') for j in jobs.values()):status('blocked');return
            # Give finished jobs' slots to a survivor, preserving global batch and
            # sample ordinals through an explicitly recorded checkpoint migration.
            used={s for j in jobs.values() if j['status'] not in ('complete','failed') for s in j.get('expanding_to',j['slots'])};free=sorted(set(range(4))-used)
            if free and not any('expanding_to' in j for j in jobs.values()):
                candidates=[(arm,j) for arm,j in jobs.items() if arm in active and len(j['slots'])<4]
                candidates.sort(key=lambda item: read(Path(read(item[1]['config'])['output'])/'status.json').get('step',0) if (Path(read(item[1]['config'])['output'])/'status.json').exists() else 0)
                for arm,job in candidates:
                    n=len(job['slots']);growth=n if n<=len(free) else 0
                    if n==1 and len(free)==3:growth=3
                    if growth:
                        job['expanding_to']=sorted(job['slots']+free[:growth]);persist();(Path(read(job['config'])['output'])/'STOP').write_text('Save a complete checkpoint for automatic GPU expansion.\n');break
            status();time.sleep(3)
    finally:
        for work in active.values():work['cancel'].set()
        for work in active.values():work['thread'].join(timeout=120)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve();q=root/'parallel_queue';q.mkdir(exist_ok=True)
    with (q/'controller.lock').open('a') as lock:
        while True:
            if (q/'STOP').exists():return
            if (q/'state.json').exists() and read(q/'state.json').get('status')=='complete':return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:run(root);return
            except Exception as e:atomic_json(q/'state.json',dict(status='blocked',error=repr(e),host=socket.gethostname(),at=time.time()));raise
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':main()
