"""Single-GPU DAG workers; old production recovery/expansion has priority."""
from pathlib import Path
import copy,fcntl,json,os,sys,threading,time,socket
from p2_listen.contracts import ROOT,read,atomic_json
sys.path.insert(0,str(ROOT/'scripts'))
from listen_cluster import launch
from p2_legacy_qwen_queue import resources


def old_reservations(root):
    path=Path(root)/'parallel_queue/jobs.json'
    if not path.exists():return {},set()
    jobs=read(path);active={};priority=set()
    for arm,job in jobs.items():
        if job['status'] in ('complete','failed'):continue
        slots=job.get('expanding_to',job['slots'])
        gpus={g for s in slots for g in (2*s,2*s+1)}
        for g in gpus:active[g]=arm
        if 'expanding_to' in job or job['status']!='running':priority|=gpus
    return active,priority


def worker_command(task,host,gpu,shared,attempt,root):
    spec=read(root/'pipeline.json');py=spec['python'];kind=task['kind']
    env=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=4','PYTHONUNBUFFERED=1','TOKENIZERS_PARALLELISM=false','PYTORCH_ALLOC_CONF=expandable_segments:True']
    if kind=='semantic':
        memory=.46 if shared else .80;batch=8 if shared else 16
        cmd=env+['LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6',spec['vllm_python'],str(ROOT/'scripts/run_listener_semantic_prompt.py'),'--inputs',task['inputs'],'--output',task['output'],'--model',spec['semantic_model'],'--batch',str(batch),'--memory',str(memory),'--attempts','3']
    elif kind in ('text','input_plan'):
        cmd=env+['P2_PLAN_GPU_MEMORY_GIB=64',py,str(ROOT/'scripts/p2_plan_cache.py'),'cache-text' if kind=='text' else 'cache-input-plan','--work',task['work'],'--shard-index',str(task.get('shard_index',0)),'--shards',str(task.get('shards',1)),'--device','cuda:0']
    elif kind=='s0':
        cmd=env+['P2_TRAIN_MEMORY_GIB=64',py,str(ROOT/'scripts/p2_guided_chain_s0.py'),'--root',str(root),'--chunk',task['chunk'],'--output',task['output']]
    else:raise ValueError('Unknown single-GPU task')
    (root/'locks').mkdir(exist_ok=True)
    return ['flock','-n',str(root/'locks'/(task['id']+'.lock')),*cmd]


def run(root):
    spec=read(root/'pipeline.json');cluster=read(root/'cluster.json');out=root/'pool';out.mkdir(exist_ok=True);running={};states={}
    for p in (out/'tasks').glob('*.json') if (out/'tasks').exists() else []:
        states[p.stem]=read(p)
        if states[p.stem]['status']=='running':states[p.stem]['status']='pending'
    def write(tid,value):states[tid]=value;atomic_json(out/'tasks'/(tid+'.json'),value)
    inventory=[];last_probe=0
    try:
        while not (root/'STOP').exists() and not (root/'TRAINING_READY.json').exists():
            now=time.time();legacy,priority=old_reservations(spec['old_training'])
            controls=read(root/'resource_controls.json') if (root/'resource_controls.json').exists() else {}
            max_workers=controls.get('max_workers',24)
            for tid,work in list(running.items()):
                if work['gpu'] in priority:work['preempted']='old_training_recovery_priority';work['cancel'].set()
                elif states[tid].get('shared_with_old_training') and work['gpu'] not in legacy:
                    work['preempted']='upgrade_to_exclusive_gpu_budget';work['cancel'].set()
                stamps=[work['started_at']]+[p.stat().st_mtime for p in work['progress_paths'] if p.exists()]
                if now-max(stamps)>1200:work['cancel'].set()
                if work['thread'].is_alive():continue
                work['thread'].join();del running[tid];state=states[tid]
                if work.get('preempted'):
                    write(tid,dict(state,status='pending',retry_at=now+5,reason=work['preempted']));continue
                if work['error'] is None:
                    write(tid,dict(state,status='succeeded',finished_at=now));continue
                messages='\n'.join(p.read_text(errors='replace')[-10000:] for p in work['logs'].glob('*.log'))
                data_error='Context overflow; no truncation allowed' in messages
                if data_error:write(tid,dict(state,status='data_error',finished_at=now,error=messages[-2500:]));continue
                oom=any(x in messages for x in ('out of memory','OutOfMemory','Free memory on device','No available memory for the cache blocks'))
                write(tid,dict(state,status='pending',retry_at=now+min(300,15*state['attempt']),exclusive_only=state.get('exclusive_only',False) or oom,error=work['error'],last_error_log=str(work['logs'])))
            if now-last_probe>5:
                try:inventory=resources(cluster);last_probe=now
                except Exception as e:
                    atomic_json(out/'probe_error.json',dict(at=now,error=str(e)));inventory=[];last_probe=now
            tasks=[]
            for p in (root/'tasks').glob('*.json'):
                task=read(p);state=states.get(task['id'],dict(status='pending',attempt=0))
                if task['id'] in running or state['status'] in ('succeeded','data_error'):continue
                if state.get('retry_at',0)>now:continue
                if any(states.get(d,{}).get('status')!='succeeded' for d in task.get('depends',[])):continue
                tasks.append(task)
            tasks.sort(key=lambda t:(t.get('priority',20),t['id']))
            busy={(w['node'],w['gpu']) for w in running.values()}
            for node in cluster['nodes']:
                entry=next((x for x in inventory if x['node']==node['host']),None)
                if entry is None:continue
                for gpu,free,util in entry['gpus']:
                    if not tasks or len(running)>=max_workers:break
                    if (node['name'],gpu) in busy or gpu in priority:continue
                    shared=gpu in legacy
                    # Do not consume a GPU before its old training model has loaded.
                    if shared and free>100*1024:continue
                    candidates=[t for t in tasks if not (shared and states.get(t['id'],{}).get('exclusive_only'))]
                    if not candidates:continue
                    task=candidates[0];minimum=68 if shared or task['kind']!='semantic' else 120
                    if free<minimum*1024:continue
                    tid=task['id'];state=states.get(tid,{});attempt=state.get('attempt',0)+1
                    cmd=worker_command(task,node['host'],gpu,shared,attempt,root)
                    cc=dict(cluster,nodes=[node]);logs=out/'attempts'/f'{tid}-{attempt}-{time.time_ns()}'
                    work=dict(node=node['name'],gpu=gpu,cancel=threading.Event(),logs=logs,error=None,started_at=now,progress_paths=[Path(task.get('output',task.get('work','')))/'attempts',Path(task.get('output',task.get('work','')))/'progress.json',logs/(node['name']+'.log')])
                    def execute(cc=cc,cmd=cmd,logs=logs,work=work):
                        try:launch(cc,[cmd],logs,cancel_event=work['cancel'])
                        except Exception as e:work['error']=str(e)
                    work['thread']=threading.Thread(target=execute);running[tid]=work
                    write(tid,dict(state,status='running',attempt=attempt,node=node['name'],gpu=gpu,shared_with_old_training=shared,started_at=now,logs=str(logs),command=cmd))
                    work['thread'].start();busy.add((node['name'],gpu));tasks.remove(task)
            counts={k:sum(s['status']==k for s in states.values()) for k in ('pending','running','succeeded','data_error')}
            counts['not_started']=len(list((root/'tasks').glob('*.json')))-len(states)
            atomic_json(out/'state.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),counts=counts,running={k:dict(node=v['node'],gpu=v['gpu']) for k,v in running.items()},legacy_priority=sorted(priority)))
            time.sleep(2)
    finally:
        for work in running.values():work['cancel'].set()
        for work in running.values():work['thread'].join(timeout=120)


def supervise(root):
    root=Path(root).resolve();(root/'pool').mkdir(exist_ok=True)
    with (root/'pool/controller.lock').open('a') as lock:
        while not (root/'STOP').exists() and not (root/'TRAINING_READY.json').exists():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:run(root)
            except Exception as e:atomic_json(root/'pool/controller_error.json',dict(at=time.time(),host=socket.gethostname(),error=repr(e)));time.sleep(10)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
