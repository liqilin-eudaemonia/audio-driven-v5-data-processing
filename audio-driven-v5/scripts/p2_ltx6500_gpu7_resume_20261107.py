"""Resume only physical node11 GPU7; keep all committed6500 results and metrics."""
import argparse,copy,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
WORK=ROOT/'experiments/p2_ltx6500_latest_sft_dual200_20261107';WORKER=ROOT/'scripts/p2_ltx6500_latest_sft_dual200_20261107.py'


def live_owned(pid,script):
    try:
        p=Path('/proc',str(pid));s=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return s[2]!='Z' and str(script) in cmd and str(WORK) in cmd
    except FileNotFoundError:return False


def prepare(work):
    out=work/'card7';out.mkdir(exist_ok=True)
    if (out/'PREPARED.json').exists():return read(work/'protocol.json')
    original=read(work/'CONTROLLER.json')
    if live_owned(original['pid'],ROOT/'scripts/p2_ltx6500_overlap_20261107.py'):raise RuntimeError('Old controller still live; finish its prior STOP first')
    for receipt in (work/'overlap/processes/render').glob('*.json'):
        record=read(receipt)
        if live_owned(record['pid'],WORKER):raise RuntimeError('An old multi-card worker still live; preserve STOP until exit')
    p=read(work/'protocol.json');save(out/'original_protocol.json',p);save(out/'ORIGINAL_CONTROLLER.json',original)
    cluster=read(work/'cluster.json');save(out/'original_cluster.json',cluster)
    cluster['nodes'][0]['gpus']=[7];cluster['training_script']=str(SELF)
    updated=copy.deepcopy(p);updated['execution_code'][str(SELF)]=sha(SELF)
    updated['authorized_scope']='Latest user authorization: node11 physical GPU7 only; resume6500, leave GPU0..6 for other work'
    updated['execution_order']='singleGPU7 remaining render -> decode -> same8 scoring shards onGPU7 -> full200 aggregate'
    assert all(updated[k]==p[k] for k in ['arms','checkpoint_identity','datasets','seed','generation','planner_adapter_identity','shared_items_sha256','workers'])
    stop=work/'STOP'
    if stop.exists():
        raw=stop.read_text();(out/'PREVIOUS_STOP.txt').write_text(raw)
        # The latest explicit user instruction resumes this evaluation onGPU7;
        # it supersedes its previous all-card pause, not any training STOP.
        if stop.read_text()!=raw:raise RuntimeError('STOP changed concurrently')
        stop.unlink()
    save(work/'cluster.json',cluster);save(work/'protocol.json',updated)
    items=read(work/'shared_items.json');arm=p['arms'][0]['name']
    retained=[x['key'] for x in items if (work/'latents'/arm/x['key']/'complete.json').exists()]
    save(out/'PREPARED.json',dict(success=True,at=time.time(),physical_gpu=7,retained_latents=len(retained),retained_keys=retained,latest_user_resumed_only_this_evaluation=True,previous_stop_archived=True,old_multi_gpu_workers_exited=True))
    return updated


def state(work,stage,**kw):
    value=dict(stage=stage,physical_gpu=7,at=time.time(),**kw);save(work/'status.json',value);print(json.dumps(value),flush=True)


def phase(work,stage):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');items=read(work/'shared_items.json');arm=p['arms'][0]['name'];folder=work/('latents' if stage=='render' else 'generated')/arm
    if all(ready(folder/x['key']) for x in items):return True
    state(work,stage,completed=sum(ready(folder/x['key']) for x in items),total_outputs=400)
    for attempt in range(1,4):
        if (work/'STOP').exists():return False
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
        for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','P2_TRAIN_MEMORY_GIB']:env.pop(k,None)
        log=work/'card7'/f'{stage}_attempt{attempt}.log'
        with log.open('ab',buffering=0) as f:
            proc=subprocess.Popen([str(PYTHON),'-u',str(WORKER),'worker','--work',str(work),'--stage',stage,'--rank','7'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            save(work/'card7'/f'{stage}_process.json',dict(pid=proc.pid,physical_gpu=7,rank=7,at=time.time(),log=str(log)))
            code=proc.wait()
        if (work/'STOP').exists():return False
        if code==0 and all(ready(folder/x['key']) for x in items):
            save(work/'stages'/(stage+'.json'),dict(complete=True,physical_gpu=7,at=time.time(),preserved_existing=True));return True
        time.sleep(5)
    raise RuntimeError('SingleGPU7 phase failed: '+stage)


def run(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Run only on node11')
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare(work)
        from p2_speaker_whisper_after_annotation import process_record
        save(work/'CONTROLLER.json',process_record());save(work/'card7/CONTROLLER.json',process_record())
        for part in ['render','decode']:
            if not phase(work,part):state(work,'stopped');return
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,ltx_step=6500,physical_gpu=7,at=time.time()))
        from p2_testsets_eval.cluster import bind
        bind(work)
        import p2_testsets_eval.metrics as scoring
        import listen_cluster
        save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=400,ltx_step=6500,planner_step=2752,at=time.time()))
        scoring.prepare_smoke(work)
        original_transport=listen_cluster.transport
        def transport(cfg,node,cmd,**kwargs):
            return original_transport(cfg,node,['env','CUDA_VISIBLE_DEVICES=7',*cmd],**kwargs)
        def headroom(report):
            # User permits shared use of GPU7. Native idle-only detection would
            # otherwise wait despite sufficient memory. No other GPU is eligible.
            result=[]
            for line in report.splitlines():
                index,used,total,util=[float(x.strip()) for x in line.split(',')]
                if int(index)==7 and total-used>=40*1024:result.append(7)
            return result
        state(work,'metrics')
        with patch.object(listen_cluster,'transport',transport),patch.object(scoring,'idle_gpus',headroom):scoring.queue(work)
        if (work/'STOP').exists():state(work,'stopped');return
        import p2_ltx6500_latest_sft_dual200_20261107 as target
        target.final(work,p);save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,ltx_step=6500,planner_step=2752,physical_gpu=7,at=time.time()));state(work,'complete',outputs=400)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run']);parser.add_argument('--work',type=Path,default=WORK);a=parser.parse_args()
    try:run(a.work)
    except Exception as exc:save(a.work/'card7/FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
