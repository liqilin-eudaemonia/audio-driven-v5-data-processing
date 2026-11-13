"""Overlap groups and CPU/event/video stages; leases protect each physical GPU."""
from pathlib import Path
import argparse,concurrent.futures,copy,fcntl,inspect,json,os,subprocess,sys,time,traceback
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1];LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(LEGACY),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON
import driver as D
import parallel_events as original
from rolling_video import acquire

def validate():
    plan=read(WORK/'ROLLING_PLAN.json')
    for f,h in plan['source'].items():
        if sha(f)!=h:raise ValueError('Rolling dispatch source changed')

def leased_node(events,index):
    validate();cluster=read(WORK/'PLAN.json')['cluster'];n=cluster['nodes'][index]
    if n['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong event node')
    waiting=WORK/'event_waiting';waiting.mkdir(exist_ok=True);tag=__import__('hashlib').sha256(str(events).encode()).hexdigest()[:16]
    pending=[]
    for gpu in n['gpus']:
        rank=index*8+gpu;out=events/'inference_shards'/f'{rank:02d}'
        if(out/'DETECTION_COMPLETE.json').exists():
            done=read(out/'DETECTION_COMPLETE.json');contract=read(out/'detector_contract.json')
            if done['detector_identity']!=contract['identity_hash']:raise ValueError('Cached detector identity changed')
            from p2_event_presence.contracts import rows
            media={m['media_id']:m for r in rows(out/'prepared_pairs.jsonl')for m in r['prepared_media'].values()}
            for mid,h in done['detections'].items():
                if sha(Path(media[mid]['folder'])/'detection.json')!=h:raise ValueError('Cached detection changed')
            continue
        pending.append(rank)
    active={};streams=[]
    for rank in pending:save(waiting/f'{rank:02d}_{tag}.json',dict(work=str(events),rank=rank,at=time.time()))
    try:
        while pending or active:
            if(WORK/'STOP').exists():raise RuntimeError('User STOP')
            for rank,job in list(active.items()):
                rc=job['process'].poll()
                if rc is None:continue
                job['lease'].close();del active[rank]
                if rc:raise RuntimeError(f'Event rank{rank} failed: '+str(job['log']))
            for rank in list(pending):
                lease=acquire(rank)
                if lease is None:continue
                gpu=rank%8;out=events/'inference_shards'/f'{rank:02d}'
                try:
                    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                    save(out/'EXECUTION.json',dict(host=n['host'],physical_gpu=gpu,uuid=uuid,rank=rank,source_sha256=sha(Path(original.__file__))),immutable=True)
                    cfg=read(events/'contract.json')['policy'];py=cfg['inference_python']
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1')
                    env['LD_LIBRARY_PATH']=str(Path(py).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
                    log=out/'inference.log';stream=log.open('ab',buffering=0);streams.append(stream)
                    p=subprocess.Popen([py,'-u',str(Path(original.__file__)),'infer','--work',str(out)],env=env,stdout=stream,stderr=subprocess.STDOUT)
                    active[rank]=dict(process=p,lease=lease,log=log);pending.remove(rank)
                    (waiting/f'{rank:02d}_{tag}.json').unlink(missing_ok=True)
                    save(out/'GPU_LEASE_DISPATCH.json',dict(pid=p.pid,rank=rank,uuid=uuid,dispatcher_sha256=sha(Path(__file__)),at=time.time()))
                except BaseException:lease.close();raise
            time.sleep(.5)
    finally:
        for job in active.values():
            if job['process'].poll()is None:job['process'].terminate()
        for job in active.values():
            try:job['process'].wait(timeout=15)
            except subprocess.TimeoutExpired:job['process'].kill();job['process'].wait()
            job['lease'].close()
        for rank in pending:(waiting/f'{rank:02d}_{tag}.json').unlink(missing_ok=True)
        for f in streams:f.close()

def event_run(work):
    validate()
    # Existing detector sharding/identity merge, only the node dispatcher changes.
    source=inspect.getsource(original.parallel_infer)
    needle="'node','--work'"
    if source.count(needle)!=1:raise ValueError('Event dispatcher changed')
    source=source.replace(needle,"'leased-node','--work'")
    namespace=dict(original.__dict__,__file__=str(Path(__file__)))
    exec(compile(source,str(Path(__file__))+'::leased_event_dispatch','exec'),namespace)
    def cpu(events,cluster,stage):
        # One28-worker CPU batch at a time, overlapped with all GPU stages.
        with(WORK/'event_cpu.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            original.parallel_cpu(events,cluster,stage)
    function=original.stage_adapter();function.__globals__['parallel_cpu']=cpu
    function.__globals__['parallel_infer']=namespace['parallel_infer']
    function(work)

def score_events(work):
    from listen_cluster import transport
    cl=read(WORK/'PLAN.json')['cluster'];guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
    argv=[str(PYTHON),str(guard),'--guard',str(PYTHON),'-u',str(Path(__file__)),'events','--work',str(work)]
    with(work/'rolling_events.log').open('ab',buffering=0)as log:
        process=subprocess.Popen(transport(cl,cl['nodes'][0],argv),cwd=ROOT,stdin=subprocess.PIPE,
                                 stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll()is None:
                if(WORK/'STOP').exists()or(WORK/'ROLLING_ABORT.json').exists():
                    process.stdin.close();raise RuntimeError('Rolling event stage cancelled')
                time.sleep(2)
            if process.returncode:raise RuntimeError('Event group failed: '+str(work/'rolling_events.log'))
        finally:
            if process.stdin and not process.stdin.closed:process.stdin.close()
    D.seal_score(work)

def run():
    validate()
    from p2_loss_search28.resources import idle
    from stage_driver import verify_scoring
    plan=read(WORK/'PLAN.json')
    if not read(WORK/'TRAINING_PAUSED_ACCEPTED.json')['success']or not idle(plan['cluster'])['idle']:
        raise RuntimeError('Handoff requires released allocation')
    with(WORK/'rolling_controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,dispatcher='rolling',at=time.time()))
        guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
        log=(WORK/'rolling_video.log').open('ab',buffering=0)
        video=subprocess.Popen([str(PYTHON),str(guard),'--guard',str(PYTHON),'-u',str(WORK/'rolling_video.py')],
                               cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        pending=[D.folder(n,s)for n in D.NAMES for s in D.SEEDS];active={};pool=concurrent.futures.ThreadPoolExecutor(2)
        success=False
        try:
            while pending or active:
                if(WORK/'STOP').exists():raise RuntimeError('User STOP')
                if video.poll()not in(None,0):raise RuntimeError('Rolling video failed; see rolling_video.log')
                for future,work in list(active.items()):
                    if future.done():future.result();del active[future]
                for work in list(pending):
                    if(work/'SCORING_COMPLETE.json').exists():verify_scoring(work);pending.remove(work);continue
                    if len(active)>=2:break
                    if not(work/'FINAL_AUDIT.json').exists():continue
                    active[pool.submit(score_events,work)]=work;pending.remove(work)
                save(WORK/'STATUS.json',dict(stage='rolling_multi_group_evaluation',at=time.time(),
                     video_active=video.poll()is None,event_groups=[p.name for p in active.values()],
                     pending_groups=[p.name for p in pending],training_resumed=False))
                time.sleep(2)
            if video.wait()!=0:raise RuntimeError('Rolling video failed')
            D.report()
            result=read(WORK/'COMPARISON.json');result['rolling_dispatch_plan_sha256']=sha(WORK/'ROLLING_PLAN.json');save(WORK/'COMPARISON.json',result)
            save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=512,training_remains_stopped=True));success=True
            save(WORK/'STATUS.json',dict(stage='awaiting_experiment_direction',at=time.time(),training_resumed=False))
        finally:
            if not success:save(WORK/'ROLLING_ABORT.json',dict(at=time.time(),reason='Stop owned sibling tasks after controller stop/failure'))
            if video.stdin and not video.stdin.closed:video.stdin.close()
            try:video.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            pool.shutdown(wait=False,cancel_futures=True);log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','events','leased-node']);p.add_argument('--work',type=Path);p.add_argument('--index',type=int);a=p.parse_args()
    try:
        if a.action=='run':run()
        elif a.action=='events':event_run(a.work.resolve())
        else:leased_node(a.work.resolve(),a.index)
    except Exception as exc:save(WORK/'ROLLING_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
