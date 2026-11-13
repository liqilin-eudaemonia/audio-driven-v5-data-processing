"""Adopt the live GPU7 worker; add seven GPUs to the same immutable dual200 queue."""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
WORK=ROOT/'experiments/p2_event_whisper6309_dual200_node11_20261107'
OLD=ROOT/'scripts/p2_event_whisper6309_eval_node11_20261107.py'


def gpu(rank):return 7 if rank==0 else rank-1


def process(pid):
    p=Path('/proc',str(pid));s=(p/'stat').read_text().split()
    return dict(pid=int(pid),start_ticks=s[21],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def live(record):
    try:
        p=Path('/proc',str(record['pid']));s=(p/'stat').read_text().split()
        return s[21]==record['start_ticks'] and s[2]!='Z'
    except FileNotFoundError:return False


def worker(work,stage,rank):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu(rank)):raise ValueError('Unexpected physical GPU mapping')
    import p2_event_whisper6309_eval_node11_20261107 as base
    with base.runtime():
        import torch
        if torch.cuda.device_count()!=1:raise ValueError('Expected exactly one visible GPU')
        save(work/'expansion8/devices'/f'rank{rank:02d}.json',dict(rank=rank,physical_gpu=gpu(rank),pid=os.getpid(),host=os.uname().nodename,stage=stage,cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),at=time.time()))
        from p2_testsets_eval.whisper_runtime import worker as native
        native(work,stage,rank)


class Adopted:
    def __init__(self,record,marker):self.record=record;self.marker=marker;self.pid=record['pid'];self.returncode=None
    def poll(self):
        if self.returncode is not None:return self.returncode
        if live(self.record):return None
        self.returncode=0 if self.marker.exists() and read(self.marker).get('status')=='complete' else 1
        return self.returncode


def state(work,stage,**kw):
    value=dict(stage=stage,at=time.time(),**kw);save(work/'status.json',value);save(work/'expansion8/state.json',value);print(json.dumps(value),flush=True)


def stage(work,name,adopt=None):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');items=read(work/'shared_items.json');arm=p['arms'][0]['name'];active={};attempts={};files={};done={};next_report=0
    folder=work/('latents' if name=='render' else 'generated')/arm
    def count():return sum((folder/x['key']/'complete.json').exists() for x in items)
    def launch(rank):
        attempts[rank]=attempts.get(rank,0)+1
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu(rank)),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True')
        for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','P2_TRAIN_MEMORY_GIB']:env.pop(k,None)
        path=work/'expansion8/logs'/f'{name}_rank{rank:02d}_attempt{attempts[rank]}.log';path.parent.mkdir(parents=True,exist_ok=True)
        stream=path.open('ab',buffering=0);files[rank]=stream
        proc=subprocess.Popen([str(PYTHON),'-u',str(SELF),'worker','--work',str(work),'--stage',name,'--rank',str(rank)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        active[rank]=proc;save(work/'expansion8/processes'/name/f'{rank:02d}.json',dict(**process(proc.pid),physical_gpu=gpu(rank),rank=rank,log=str(path),attempt=attempts[rank],at=time.time()))
    if adopt and live(adopt):
        active[0]=Adopted(adopt,work/'progress'/name/'00.json');attempts[0]=1
        save(work/'expansion8/processes'/name/'00.json',dict(**adopt,physical_gpu=7,rank=0,adopted=True))
    else:launch(0)
    for rank in range(1,8):launch(rank);time.sleep(.4)
    try:
        while active:
            for rank,proc in list(active.items()):
                code=proc.poll()
                if code is None:continue
                del active[rank]
                if rank in files:files.pop(rank).close()
                if code:
                    if attempts[rank]>=3:raise RuntimeError(f'{name} rank{rank} failed after3 attempts')
                    launch(rank)
                else:done[rank]=dict(rank=rank,physical_gpu=gpu(rank),returncode=0,attempts=attempts[rank])
            if time.time()>=next_report:
                state(work,name,completed=count(),total_outputs=400,active_ranks=sorted(active),gpus=[gpu(r) for r in sorted(active)],shared_queue=True)
                next_report=time.time()+20
            time.sleep(2)
        if (work/'STOP').exists():state(work,'stopped');return False
        assert all(ready(folder/x['key']) for x in items),'Incomplete shared queue after all workers exit'
        save(work/'stages'/(name+'.json'),dict(complete=True,world=8,results=list(done.values()),at=time.time(),preserved_existing=True))
        return True
    finally:
        # Do not terminate any unrelated task. In a controller exception, our
        # current sample processes may finish and commit for a later resume.
        for stream in files.values():stream.close()


def prepare(work):
    out=work/'expansion8';out.mkdir(exist_ok=True)
    if (out/'PREPARED.json').exists():
        p=read(work/'protocol.json');assert p['execution_code'][str(SELF)]==sha(SELF)
        return p
    oldp=read(work/'protocol.json');oldcluster=read(work/'cluster.json')
    assert oldp['workers']==8 and oldp['shared_item_queue'] is True
    assert oldcluster['nodes'][0]['host']=='172.29.189.119' and oldcluster['nodes'][0]['gpus']==[7]
    assert not (work/'STOP').exists()
    original=read(work/'CONTROLLER.json');worker_info=read(work/'device.json');assert worker_info['stage']=='render'
    old_worker=process(worker_info['pid']);controller=process(original['pid'])
    assert controller['start_ticks']==original['start_ticks'] and str(OLD)+' run' in controller['command']
    assert str(OLD)+' worker' in old_worker['command'] and '--stage render' in old_worker['command']
    save(out/'ORIGINAL_CONTROLLER.json',original);save(out/'ORIGINAL_WORKER.json',old_worker)
    save(out/'original_protocol.json',oldp);save(out/'original_cluster.json',oldcluster)
    # Kill the original controller PID only, never its process group or worker.
    # This prevents it advancing to decode/bind before assisting ranks finish.
    os.kill(original['pid'],signal.SIGTERM)
    deadline=time.time()+15
    while live(original):
        if time.time()>deadline:raise TimeoutError('Old controller still live; leave worker untouched')
        time.sleep(.2)
    assert live(old_worker),'Adopted GPU7 worker unexpectedly exited during handoff'
    p=copy.deepcopy(oldp);p['worker_cli']=str(SELF);p['authorized_scope']='node11 physical GPUs0..7; keep107+ existing outputs; no eviction of unrelated tasks'
    p['execution_code'][str(SELF)]=sha(SELF);p['metrics_workers_per_gpu']=4
    c=copy.deepcopy(oldcluster);c['nodes'][0]['gpus']=list(range(8));c['training_script']=str(SELF)
    save(work/'cluster.json',c);save(work/'protocol.json',p)
    frozen=['arms','datasets','seed','generation','checkpoint_identity','shared_items_manifest','shared_items_sha256','total_samples','total_outputs','sampling','planner','whisper_encoder','feature_code']
    assert all(oldp[k]==p[k] for k in frozen)
    from p2_speaker_whisper_after_annotation import process_record
    save(work/'CONTROLLER.json',process_record())
    save(out/'PREPARED.json',dict(success=True,at=time.time(),adopted=old_worker,old_controller_stopped=True,worker_untouched=True,science_unchanged=frozen,rank_gpu={r:gpu(r) for r in range(8)}))
    return p


def run(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Run only on node11')
    work=work.resolve();out=work/'expansion8';out.mkdir(exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=prepare(work)
        # The old controller has exited and its lock is free. Own it now.
        with (work/'controller.lock').open('a') as parent_lock:
            fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from p2_speaker_whisper_after_annotation import process_record
            save(work/'CONTROLLER.json',process_record());save(out/'CONTROLLER.json',process_record())
            import p2_event_whisper6309_eval_node11_20261107 as base
            for part in ['render','decode']:
                if (work/'stages'/(part+'.json')).exists():continue
                adopted=read(out/'ORIGINAL_WORKER.json') if part=='render' else None
                if not stage(work,part,adopted):return
            from p2_testsets_eval.runtime import ready
            assert all(ready(work/'generated'/base.ARM/x['key']) for x in read(work/'shared_items.json'))
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time(),inference_gpus=8))
            state(work,'binding')
            from p2_testsets_eval.cluster import bind
            bind(work);base.smoke_gate(work,p)
            state(work,'metrics',gpus=list(range(8)))
            from p2_testsets_eval.metrics import queue
            queue(work)
            if (work/'STOP').exists():state(work,'stopped');return
            base.final_audit(work,p)
            save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,at=time.time(),training_resumed=False,inference_gpus=8))
            state(work,'complete',outputs=400)


def launch(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Launch on node11 only')
    # Seven newly allocated cards must be free; GPU7 is explicitly shared.
    report=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values={int(v[0]):[float(x) for x in v[1:]] for line in report.splitlines() if (v:=line.split(','))}
    assert all(values[g][0]<2048 and values[g][1]<10 for g in range(7)),values
    out=work/'expansion8';out.mkdir(exist_ok=True)
    if (out/'CONTROLLER.json').exists() and live(read(out/'CONTROLLER.json')):raise RuntimeError('Eight-GPU controller already running')
    save(out/'BEFORE_RESOURCES.json',dict(csv=report,at=time.time()))
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (out/'controller.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([str(PYTHON),'-u',str(SELF),'run','--work',str(work)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'LAUNCH.json',dict(pid=proc.pid,at=time.time(),gpus=list(range(8))));print('expansion controller launched',proc.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','run','worker']);parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage',choices=['render','decode']);parser.add_argument('--rank',type=int);a=parser.parse_args()
    try:
        if a.action=='launch':launch(a.work)
        elif a.action=='worker':worker(a.work,a.stage,a.rank)
        else:run(a.work)
    except Exception as exc:
        save(a.work/'expansion8/FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
