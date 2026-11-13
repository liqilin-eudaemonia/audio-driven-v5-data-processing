"""Durable semantic -> cache -> common view -> paired training transitions."""
from pathlib import Path
import fcntl,json,time,os,socket,traceback
from p2_listen.contracts import ROOT,read,atomic_json,sha
from . import backend  # Load package before legacy audit inserts scripts/ into sys.path.
from .data import export_chunk,publish_task,finish_view


def process_semantic(root,task,state):
    folder=Path(task['inputs']).parent
    if (folder/'TRANSITION.json').exists():return
    if (folder/'export.json').exists():
        export=read(folder/'export.json');wanted=set(export['unresolved_ids'])
        unresolved=[json.loads(l) for l in Path(task['inputs']).read_text().splitlines() if l.strip() and json.loads(l)['job_id'] in wanted]
    else:export,unresolved=export_chunk(root,task)
    for kind in ('text','input_plan'):
        if export['accepted']:publish_task(root,dict(id=task['id']+'_'+kind,kind=kind,work=export['work'],priority=10))
    if unresolved:
        if state['status'] not in ('data_error','replay_conflict'):raise ValueError('Successful native worker omitted terminal outcomes')
        if len(unresolved)==1 and state['status']=='data_error':
            job=unresolved[0]
            atomic_json(root/'context_rejections'/(job['job_id']+'.json'),dict(sample_id=job['record']['sample_id'],job_id=job['job_id'],stage='native_context_overflow',error=state['error'],task=task['id'],inputs=task['inputs']),immutable=True)
        else:
            from .partition import groups
            partitions=([unresolved[i:i+16] for i in range(0,len(unresolved),16)] if state['status']=='replay_conflict' else groups(unresolved,read(root/'pipeline.json')['semantic_model']))
            for index,jobs in enumerate(partitions):
                suffix=f'p{index:03d}'
                tid=task['id']+suffix;child=root/'chunks'/tid;child.mkdir(exist_ok=True)
                inputs=child/'inputs.jsonl';content=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
                if inputs.exists() and inputs.read_text()!=content:raise ValueError('Child input changed')
                inputs.write_text(content)
                publish_task(root,dict(id=tid,kind='semantic',inputs=str(inputs),output=str(child/'run'),priority=15))
    atomic_json(folder/'TRANSITION.json',dict(export_sha256=sha(folder/'export.json'),state=state['status']),immutable=True)


def tick(root):
    if (root/'SOURCE_LOCK.json').exists():
        for path,expected in read(root/'SOURCE_LOCK.json').items():
            if sha(path)!=expected:raise ValueError('Frozen pipeline source changed: '+path)
    tasks=[read(p) for p in sorted((root/'tasks').glob('*.json'))]
    semantic=[t for t in tasks if t['kind']=='semantic']
    for task in semantic:
        path=root/'pool/tasks'/(task['id']+'.json')
        if path.exists():
            state=read(path)
            if state['status'] in ('succeeded','data_error','replay_conflict'):process_semantic(root,task,state)
    # Newly split tasks must also be terminal before freezing membership.
    semantic=[read(p) for p in (root/'tasks').glob('*.json') if read(p)['kind']=='semantic']
    all_done=all((Path(t['inputs']).parent/'TRANSITION.json').exists() for t in semantic)
    if all_done and not (root/'view/COMPLETE.json').exists():finish_view(root)
    cached=[read(p) for p in (root/'tasks').glob('*.json') if read(p)['kind'] in ('text','input_plan')]
    caches_done=all((root/'pool/tasks'/(t['id']+'.json')).exists() and read(root/'pool/tasks'/(t['id']+'.json'))['status']=='succeeded' for t in cached)
    if not (root/'tasks/s0_canary.json').exists():
        for export_path in sorted((root/'chunks').glob('*/export.json')):
            export=read(export_path)
            if not export['accepted']:continue
            pair=[root/'pool/tasks'/(export['chunk']+'_'+kind+'.json') for kind in ('text','input_plan')]
            if all(path.exists() and read(path)['status']=='succeeded' for path in pair):
                from .canary import make
                canary=make(root,export)
                publish_task(root,dict(id='s0_canary',kind='s0',chunk=str(canary),output=str(canary/'s0'),priority=0))
                break
    if all_done and caches_done:
        from .prepare import prepare
        prepare(root)
        publish_task(root,dict(id='s0_final',kind='s0',chunk='final',output=str(root/'s0'),priority=0))
        status=root/'pool/tasks/s0_final.json'
        if status.exists() and read(status)['status']=='succeeded':
            report=read(root/'s0/COMPLETE.json')
            if report.get('success') is not True:raise ValueError('S0 did not pass')
            atomic_json(root/'TRAINING_READY.json',dict(samples=read(root/'view/COMPLETE.json')['samples'],s0_sha256=sha(root/'s0/COMPLETE.json'),configs_sha256=sha(root/'configs/matrix.json')),immutable=True)
    atomic_json(root/'director/status.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),semantic_tasks=len(semantic),semantic_done=sum((Path(t['inputs']).parent/'TRANSITION.json').exists() for t in semantic),cache_tasks=len(cached),all_semantic_done=all_done,all_cache_done=caches_done,training_ready=(root/'TRAINING_READY.json').exists()))


def supervise(root):
    root=Path(root).resolve();(root/'director').mkdir(exist_ok=True)
    with (root/'director/controller.lock').open('a') as lock:
        while not (root/'STOP').exists():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:
                while not (root/'STOP').exists():
                    if (root/'TRAINING_READY.json').exists():
                        from .training_queue import supervise as train
                        train(root);return
                    try:tick(root)
                    except Exception as exc:
                        atomic_json(root/'director/error.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
                    time.sleep(10)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
