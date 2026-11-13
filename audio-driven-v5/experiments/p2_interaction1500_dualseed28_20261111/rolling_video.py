"""One cross-checkpoint GPU pool using the existing audited workers and binder."""
from pathlib import Path
import concurrent.futures,fcntl,json,os,subprocess,sys,time,traceback
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1];LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse'),str(LEGACY),str(WORK)]
sys.path.insert(0,str(WORK))
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON,METRIC_PYTHON
from p2_progressive_eval_20261108 import ResourcePoller
from binder import ShardBinder
import evaluate28 as native
import eval_runtime as E
import driver as D
native.transport=__import__('listen_cluster').transport

def acquire(rank):
    folder=WORK/'gpu_leases';folder.mkdir(exist_ok=True);f=(folder/f'{rank:02d}.lock').open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
    except BlockingIOError:f.close();return None

def event_waiting(rank):return any((WORK/'event_waiting').glob(f'{rank:02d}_*.json'))

def choose(contexts,active,physical,decode_limit=4,metric_limit=2):
    """Ready work across all groups; no completed group/rank barrier."""
    keys={j['key']for j in active};jobs=[j for j in active if j['physical']==physical]
    if any(j['stage']=='render'for j in jobs):return None
    for c in contexts:
        if c['done']or c['finalizing']:continue
        tag=c['tag'];b=c['binder']
        if not c['smoke']:
            key=(tag,'smoke')
            if not jobs and key not in keys and b.smoke():return c,'metric',0,True,0,key
        elif not any(j['stage']=='decode'for j in jobs)and len(jobs)<metric_limit:
            for rank in sorted(b.published):
                key=(tag,'metric',rank)
                if key not in keys and not c['scored'](rank):return c,'metric',rank,False,0,key
    if not any(j['stage']=='metric'for j in jobs)and len(jobs)<decode_limit:
        for c in contexts:
            if c['done']or c['finalizing']:continue
            pending=c['latents']-len(c['binder'].committed)
            if pending<=sum(j['stage']=='decode'and j['context']is c for j in active):continue
            for slot in range(4):
                key=(c['tag'],'decode',physical,slot)
                if key not in keys:return c,'decode',None,False,slot,key
    if jobs:return None
    eligible=[c for c in contexts if not c['done']and not c['finalizing']and c['latents']<64
              and(c['tag'],'render',physical)not in keys]
    if not eligible:return None
    c=min(eligible,key=lambda c:(sum(j['stage']=='render'and j['context']is c for j in active),c['index']))
    return c,'render',None,False,0,(c['tag'],'render',physical)

def finalize(c):
    c['binder'].canonical_audit();work=c['work']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
             PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python')
    with(work/'rolling_merge.log').open('a')as log:
        subprocess.run([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(work)],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    E.final(work);save(work/'status.json',dict(stage='complete',at=time.time(),outputs=64))

def run():
    cluster=read(WORK/'PLAN.json')['cluster'];contexts=[]
    for i,(name,seed)in enumerate((n,s)for n in D.NAMES for s in D.SEEDS):
        work=D.folder(name,seed);p=read(work/'protocol.json');b=ShardBinder(work);b.refresh()
        if not(work/'FINAL_AUDIT.json').exists():b.publish_ready()
        c=dict(index=i,tag=work.name,work=work,p=p,binder=b,done=(work/'FINAL_AUDIT.json').exists(),
               finalizing=False,smoke=(work/'METRICS_SMOKE_COMPLETE.json').exists(),latents=0)
        def scored(rank,c=c):
            f=c['work']/'progress/metrics'/f"{c['p']['arms'][0]['name']}_{rank:02d}.json"
            return f.exists()and read(f).get('status')=='complete'
        c['scored']=scored;contexts.append(c)
    from listen_cluster import transport
    def probe(node):
        r=subprocess.run(transport(cluster,node,['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=20,check=True)
        return {int(v[0]):dict(free=v[2]-v[1])for line in r.stdout.splitlines()if(v:=[float(x.strip())for x in line.split(',')])}
    resources=ResourcePoller(cluster,probe);active=[];leases={};attempts={};failures={};pool=concurrent.futures.ThreadPoolExecutor(2);futures={};next_status=0
    decode_limits={r:4 for r in range(28)};metric_limits={r:2 for r in range(28)}
    try:
        while not all(c['done']for c in contexts):
            if(WORK/'STOP').exists():return
            for c in contexts:
                if c['done']or c['finalizing']:continue
                c['binder'].refresh();c['binder'].publish_ready()
                arm=c['p']['arms'][0]['name'];c['latents']=sum((c['work']/'latents'/arm/x['key']/'complete.json').exists()for x in c['binder'].items)
            for job in list(active):
                rc=job['process'].poll()
                if rc is None:continue
                job['process'].stdin.close();active.remove(job);c=job['context']
                if rc:
                    failures[job['key']]=failures.get(job['key'],0)+1
                    if 'out of memory'in job['log'].read_text(errors='replace').lower():
                        if job['stage']=='decode':decode_limits[job['physical']]=max(1,decode_limits[job['physical']]//2)
                        if job['stage']=='metric':metric_limits[job['physical']]=1
                    if failures[job['key']]>=3:raise RuntimeError('Repeated rolling worker failure: '+str(job['log']))
                elif job['smoke']:c['smoke']=True;save(c['work']/'METRICS_SMOKE_COMPLETE.json',dict(status='passed',at=time.time()))
            for physical,lease in list(leases.items()):
                if not any(j['physical']==physical for j in active):lease.close();del leases[physical]
            for c in contexts:
                if c['done']or c['finalizing']:continue
                if len(c['binder'].committed)==64 and all(c['scored'](i)for i in range(28))and not any(j['context']is c for j in active):
                    c['finalizing']=True;futures[pool.submit(finalize,c)]=c
            for future,c in list(futures.items()):
                if future.done():future.result();c['done']=True;del futures[future]
            reports=resources.snapshot();starts={i:0 for i in range(4)}
            for physical in range(28):
                report=reports.get(cluster['nodes'][physical//8]['name'],{})
                if physical%8 not in report or starts[physical//8]>=2:continue
                if physical not in leases and event_waiting(physical):continue
                task=choose(contexts,active,physical,decode_limits[physical],metric_limits[physical])
                if task is None:continue
                c,stage,logical,smoke,slot,key=task
                if report[physical%8]['free']<dict(render=80,decode=40,metric=24)[stage]*1024:continue
                if physical not in leases:
                    lease=acquire(physical)
                    if lease is None:continue
                    leases[physical]=lease
                attempts[key]=attempts.get(key,0)+1;E.ARM=c['p']['arms'][0]['name']
                job=native.launch(c['work'],cluster,stage,physical,slot,logical,smoke,attempts[key]);job.update(context=c,key=key)
                active.append(job);starts[physical//8]+=1
            if time.time()>=next_status:
                save(WORK/'ROLLING_VIDEO_STATUS.json',dict(at=time.time(),active=len(active),
                    tasks=[dict(work=j['context']['tag'],stage=j['stage'],physical=j['physical'])for j in active],
                    groups=[dict(work=c['tag'],done=c['done'],latents=c['latents'],decoded=len(c['binder'].committed),scored=sum(c['scored'](i)for i in range(28)))for c in contexts]))
                next_status=time.time()+10
            time.sleep(1)
        save(WORK/'ROLLING_VIDEO_COMPLETE.json',dict(success=True,at=time.time()))
    finally:
        for j in active:
            if j['process'].stdin and not j['process'].stdin.closed:j['process'].stdin.close()
        # Hold the allocation leases until guarded workers actually exit.
        for j in active:
            try:j['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        for f in leases.values():f.close()
        resources.close();pool.shutdown(wait=False,cancel_futures=True)

if __name__=='__main__':
    try:run()
    except Exception as exc:save(WORK/'ROLLING_VIDEO_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
