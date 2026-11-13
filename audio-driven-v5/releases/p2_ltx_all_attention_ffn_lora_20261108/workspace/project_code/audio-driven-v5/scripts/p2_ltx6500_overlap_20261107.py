"""Use released/headroom GPUs for6500 while6000 metrics continue unchanged."""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
WORK=ROOT/'experiments/p2_ltx6500_latest_sft_dual200_20261107';UPSTREAM=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
WORKER=ROOT/'scripts/p2_ltx6500_latest_sft_dual200_20261107.py'


def live(record):
    try:
        s=Path('/proc',str(record['pid']),'stat').read_text().split();return s[21]==record['start_ticks'] and s[2]!='Z'
    except FileNotFoundError:return False


def state(work,stage,**kw):
    v=dict(stage=stage,at=time.time(),**kw);save(work/'status.json',v);save(work/'overlap/state.json',v);print(json.dumps(v),flush=True)


def prepare(work):
    out=work/'overlap';out.mkdir(exist_ok=True)
    if (out/'PREPARED.json').exists():return read(work/'protocol.json')
    p=read(work/'protocol.json');original=read(work/'CONTROLLER.json');status=read(work/'status.json')
    if status['stage']!='waiting_step6000_complete':raise RuntimeError('6500 has already advanced; do not replace live workers')
    record=Path('/proc',str(original['pid']));cmd=(record/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if not live(original) or str(WORKER)+' queue' not in cmd:raise ValueError('Waiting controller identity differs')
    assert read(work/'CPU_ACCEPTANCE.json')['success'] and read(UPSTREAM/'GENERATION_COMPLETE.json')['total_outputs']==400
    assert all(sha(f)==h for f,h in p['execution_code'].items())
    save(out/'original_protocol.json',p);save(out/'ORIGINAL_CONTROLLER.json',original)
    # Only this waiting6500 controller; never stop6000 scoring or any GPU worker.
    os.kill(original['pid'],signal.SIGTERM)
    deadline=time.time()+15
    while live(original):
        if time.time()>deadline:raise TimeoutError('Old waiting controller has not exited')
        time.sleep(.2)
    updated=copy.deepcopy(p);updated['execution_code'][str(SELF)]=sha(SELF)
    updated['execution_order']='overlap6500 render/decode with6000 metrics; same400 conditions and native scoring'
    updated['authorized_scope']='node11 GPUs0..7; use sufficient free memory without stopping6000 scoring or other tasks'
    science=['arms','checkpoint_identity','datasets','seed','generation','planner_adapter_identity','shared_items_manifest','shared_items_sha256','total_samples','total_outputs']
    assert all(updated[k]==p[k] for k in science)
    save(work/'protocol.json',updated)
    save(out/'PREPARED.json',dict(success=True,at=time.time(),original_wait_controller_stopped=True,step6000_scoring_untouched=True,science_unchanged=science))
    return updated


def run_phase(work,part):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');items=read(work/'shared_items.json');arm=p['arms'][0]['name'];folder=work/('latents' if part=='render' else 'generated')/arm
    def all_done():return all(ready(folder/x['key']) for x in items)
    def job(rank):
        minimum=65*1024 if part=='render' else 28*1024
        for attempt in range(1,4):
            while True:
                if (work/'STOP').exists():return dict(rank=rank,stopped=True)
                if all_done():return dict(rank=rank,complete_before_start=True)
                values=subprocess.check_output(['nvidia-smi','-i',str(rank),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if float(values.strip())>=minimum:break
                time.sleep(5)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
            for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','P2_TRAIN_MEMORY_GIB']:env.pop(k,None)
            path=work/'overlap/logs'/f'{part}_rank{rank:02d}_attempt{attempt}.log';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('ab',buffering=0) as log:
                proc=subprocess.Popen([str(PYTHON),'-u',str(WORKER),'worker','--work',str(work),'--stage',part,'--rank',str(rank)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save(work/'overlap/processes'/part/f'{rank:02d}.json',dict(pid=proc.pid,rank=rank,physical_gpu=rank,log=str(path),at=time.time()))
                code=proc.wait()
            if code==0:return dict(rank=rank,physical_gpu=rank,returncode=0,attempts=attempt)
            time.sleep(5)
        raise RuntimeError(f'{part} rank{rank} failed; see overlap/logs')
    state(work,part,overlap_step6000_metrics=True,capacity_gpus=list(range(8)),total_outputs=400)
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(job,range(8)))
    if (work/'STOP').exists():state(work,'stopped');return False
    assert all_done(),'Missing outputs after shared workers exit'
    save(work/'stages'/(part+'.json'),dict(complete=True,results=results,at=time.time(),overlapped_step6000_metrics=True))
    return True


def run(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Run only on node11')
    out=work/'overlap';out.mkdir(exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare(work)
        with (work/'controller.lock').open('a') as mainlock:
            fcntl.flock(mainlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from p2_speaker_whisper_after_annotation import process_record
            save(work/'CONTROLLER.json',process_record());save(out/'CONTROLLER.json',process_record())
            for part in ['render','decode']:
                if not (work/'stages'/(part+'.json')).exists() and not run_phase(work,part):return
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,ltx_step=6500,at=time.time()))
            from p2_testsets_eval.cluster import bind
            bind(work)
            from p2_testsets_eval.metrics import prepare_smoke,queue
            save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=400,ltx_step=6500,planner_step=2752,at=time.time()))
            prepare_smoke(work);state(work,'metrics',ltx_step=6500);queue(work)
            if (work/'STOP').exists():state(work,'stopped');return
            # The comparison needs6000 results, but6500 generation/scoring do not.
            while not (UPSTREAM/'HANDOFF_COMPLETE.json').exists():
                if (work/'STOP').exists():state(work,'stopped');return
                state(work,'waiting_step6000_comparison_inputs');time.sleep(15)
            import p2_ltx6500_latest_sft_dual200_20261107 as target
            target.final(work,p);save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,ltx_step=6500,planner_step=2752,at=time.time()))
            state(work,'complete',outputs=400,ltx_step=6500)


def launch(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Launch only on node11')
    out=work/'overlap';out.mkdir(exist_ok=True)
    if (out/'CONTROLLER.json').exists() and live(read(out/'CONTROLLER.json')):raise RuntimeError('Overlap controller already live')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (out/'controller.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([str(PYTHON),'-u',str(SELF),'run','--work',str(work)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'LAUNCH.json',dict(pid=proc.pid,at=time.time(),gpus=list(range(8))));print('6500 early controller launched',proc.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','run']);parser.add_argument('--work',type=Path,default=WORK);a=parser.parse_args()
    try:launch(a.work) if a.action=='launch' else run(a.work)
    except Exception as exc:save(a.work/'overlap/FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
