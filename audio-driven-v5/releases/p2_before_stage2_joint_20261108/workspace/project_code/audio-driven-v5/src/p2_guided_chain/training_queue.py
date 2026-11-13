"""Serial 24-rank experiments: all four frozen-Qwen stages before paired LoRA stages."""
from pathlib import Path
import copy,fcntl,os,socket,sys,threading,time
from p2_listen.contracts import ROOT,read,sha,atomic_json
from . import backend
sys.path.insert(0,str(ROOT/'scripts'))
from p2_legacy_qwen_queue import latest,resources
from listen_cluster import launch
LOSSES=('av','av_event','av_feature','av_event_feature')


def next_loss(jobs):
    pending=[loss for loss in LOSSES if jobs[loss]['status']!='complete']
    return min(pending,key=lambda loss:(jobs[loss]['stage'],LOSSES.index(loss))) if pending else None


def commands(cluster,slots,config,resume,parent,port):
    cc=copy.deepcopy(cluster);gpus=sorted(g for s in slots for g in (2*s,2*s+1));result=[]
    if len(slots) not in (1,2,4):raise ValueError('Unsupported batch24 topology')
    for rank,node in enumerate(cc['nodes']):
        node['gpus']=gpus
        cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,gpus)),'PYTHONUNBUFFERED=1','TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cc['network'].items()],cc['python'],'-m','torch.distributed.run','--max_restarts=0','--nnodes=3','--nproc_per_node='+str(len(gpus)),'--node_rank='+str(rank),'--master_addr='+cc['nodes'][0]['host'],'--master_port='+str(port),str(ROOT/'scripts/p2_guided_chain_train.py'),'train','--config',str(config)]
        if resume:cmd+=['--resume',str(resume)]
        elif parent:cmd+=['--parent',str(parent)]
        result.append(cmd)
    return cc,result


def migration(source,config,world):
    from p2_matrix16.checkpoint import checked
    from p2_listen.cache import atomic_torch
    saved=checked(source);identity=copy.deepcopy(saved['identity']);before=copy.deepcopy(identity['experiment']);after=copy.deepcopy(config)
    before.pop('output');after.pop('output')
    if before!=after or world not in (6,12,24) or identity['world'] not in (6,12,24):raise ValueError('Migration may only change topology/output')
    if identity['source']!=backend.source_identity() or identity['data_identity']!=backend.data_identity(config):raise ValueError('Source/data changed')
    transition=dict(checkpoint=str(Path(source).resolve()),sha256=sha(source),from_world=identity['world'],to_world=world,step=saved['step'],global_batch=24,rng_rank_mapping=[i%len(saved['rank_rng']) for i in range(world)],model_and_optimizer_tensors_unmodified=True)
    identity['world']=world;identity['experiment']=config
    value=dict(saved,identity=identity,rank_rng=[saved['rank_rng'][i] for i in transition['rng_rank_mapping']]);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    path=out/Path(source).name;atomic_torch(path,value);atomic_json(path.with_suffix('.complete.json'),dict(sha256=sha(path),step=saved['step']));atomic_json(out/'run.json',identity,immutable=True);atomic_json(out/'topology_migration.json',dict(**transition,target_sha256=sha(path)))
    return path


def prepare_run(root,loss,stage,slots,source=None):
    name=f'guided_{loss}_s{stage}';c=read(root/'configs'/(name+'.json'));world=len(slots)*6;step=read(Path(source).with_suffix('.complete.json'))['step'] if source else 0
    base=root/'parallel_runs'/name/f'w{world}_from_{step:06d}';c['output']=str(base);path=root/'parallel_configs'/f'{name}_w{world}_from_{step:06d}.json';atomic_json(path,c,immutable=True)
    original=read(root/'configs'/(name+'.json'));atomic_json(base/'preflight.json',read(Path(original['output'])/'preflight.json'))
    if source and not (base/Path(source).with_suffix('.complete.json').name).exists():migration(source,c,world)
    return str(path)


def advance(root,loss,job,checkpoint):
    c=read(job['config']);budget=c['training']['steps']
    from p2_matrix16.checkpoint import checked
    if checked(checkpoint)['step']!=budget:raise ValueError('Cannot advance incomplete stage')
    job.setdefault('completed',{})[str(job['stage'])]=str(checkpoint)
    job.pop('expanding_to',None)
    if job['stage']==1:
        job.update(stage=2,parent=str(checkpoint),config=prepare_run(root,loss,2,job['slots']),status='pending',start_step=0,failures=0)
    else:job.update(status='complete',start_step=budget)


def run(root):
    q=root/'training_queue';spec=read(root/'pipeline.json');cluster=read(root/'cluster.json');active={}
    frozen=read(q/'frozen.json')
    def verify():
        for p,want in frozen.items():
            if sha(p)!=want:raise ValueError('Frozen source changed: '+p)
    jobs=read(q/'jobs.json')
    def persist():atomic_json(q/'jobs.json',jobs)
    def status(label='running',**extra):
        values={}
        for loss,j in jobs.items():
            c=read(j['config']);p=Path(c['output'])/'status.json';s=read(p) if p.exists() else {}
            values[loss]=dict(status=j['status'],stage=j['stage'],world=6*len(j['slots']),slots=j['slots'],step=s.get('step',j.get('start_step',0)),budget=c['training']['steps'],output=c['output'],completed=j.get('completed',{}),error=j.get('error'))
        atomic_json(q/'state.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),status=label,jobs=values,execution_policy='serial_24gpu_all_s1_then_s2',next_loss=next_loss(jobs),**extra))
    try:
        while not (root/'STOP').exists():
            verify()
            old=Path(spec['old_training'])/'parallel_queue/state.json'
            if not old.exists() or read(old)['status']!='complete':status('waiting_old_training');time.sleep(10);continue
            for loss,j in jobs.items():
                if loss in active:
                    work=active[loss]
                    if work['thread'].is_alive():
                        metric=Path(read(j['config'])['output'])/'metrics.jsonl'
                        if time.time()-max(work['began'],metric.stat().st_mtime if metric.exists() else 0)>1200:work['cancel'].set()
                        continue
                    work['thread'].join();del active[loss];cp,step=latest(Path(read(j['config'])['output']));j['start_step']=step
                    if cp and step==read(j['config'])['training']['steps']:advance(root,loss,j,cp)
                    elif 'expanding_to' in j:
                        if cp is None:raise RuntimeError('Expansion needs checkpoint')
                        j['slots']=j.pop('expanding_to');j['config']=prepare_run(root,loss,j['stage'],j['slots'],cp);j['status']='pending'
                    else:
                        j['failures']=j.get('failures',0)+1;j['error']=work['result'];j['status']='pending';j['retry_after']=time.time()+min(300,15*j['failures'])
                    persist()
                if j['status']=='complete' or loss in active or time.time()<j.get('retry_after',0):continue
                if active or loss!=next_loss(jobs):continue
                if j['slots']!=[0,1,2,3]:raise ValueError('Serial schedule requires all 24 GPUs')
                if 'expanding_to' in j:
                    cp,_=latest(Path(read(j['config'])['output']))
                    if cp is None:raise RuntimeError('Missing migration checkpoint')
                    j['slots']=j.pop('expanding_to');j['config']=prepare_run(root,loss,j['stage'],j['slots'],cp);persist()
                c=read(j['config']);cp,step=latest(Path(c['output']))
                if cp and step==c['training']['steps']:advance(root,loss,j,cp);persist();continue
                try:inventory=resources(cluster)
                except Exception as e:j['error']=str(e);continue
                chosen={g for s in j['slots'] for g in (2*s,2*s+1)}
                if any(g[1]<cluster['minimum_free_gib']*1024 for n in inventory for g in n['gpus'] if g[0] in chosen):continue
                out=Path(c['output'])
                if cp is None and (out/'run.json').exists():(out/'run.json').rename(out/f'run.failed_start.{time.time_ns()}.json')
                j['attempt']=j.get('attempt',0)+1;j['status']='running';j['error']=None
                cc,cmd=commands(cluster,j['slots'],j['config'],cp,j.get('parent'),32010+LOSSES.index(loss)*100+j['attempt']%90)
                work=dict(cancel=threading.Event(),result=[],began=time.time());logs=q/'attempts'/f'{loss}-s{j["stage"]}-{time.time_ns()}'
                def execute(cc=cc,cmd=cmd,work=work,logs=logs):
                    try:launch(cc,cmd,logs,cancel_event=work['cancel']);work['result'].append('success')
                    except Exception as e:work['result'].append(str(e))
                work['thread']=threading.Thread(target=execute);active[loss]=work;persist();work['thread'].start()
            if all(j['status']=='complete' for j in jobs.values()):status('complete');return
            status();time.sleep(3)
    finally:
        for work in active.values():work['cancel'].set()
        for work in active.values():work['thread'].join(timeout=120)


def supervise(root):
    root=Path(root);q=root/'training_queue';q.mkdir(exist_ok=True)
    if not (q/'jobs.json').exists():
        jobs={loss:dict(stage=1,status='pending',slots=[0,1,2,3],config=prepare_run(root,loss,1,[0,1,2,3]),completed={}) for i,loss in enumerate(LOSSES)};atomic_json(q/'jobs.json',jobs,immutable=True)
        frozen=backend.source_identity()
        for p in list(Path(__file__).parent.glob('*.py'))+[root/'pipeline.json',root/'cluster.json']+list((root/'configs').glob('*.json')):frozen[str(p)]=sha(p)
        atomic_json(q/'frozen.json',frozen,immutable=True)
    while not (root/'STOP').exists():
        if (q/'state.json').exists() and read(q/'state.json')['status']=='complete':return
        try:run(root)
        except Exception as exc:atomic_json(q/'controller_error.json',dict(at=time.time(),error=repr(exc)));time.sleep(10)
