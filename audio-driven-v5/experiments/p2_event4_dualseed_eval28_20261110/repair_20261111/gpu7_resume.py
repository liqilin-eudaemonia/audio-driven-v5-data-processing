"""Resume only unfinished event scoring on node11 physicalGPU7."""
from pathlib import Path
import argparse,concurrent.futures,copy,fcntl,inspect,json,os,signal,subprocess,sys,time,traceback
from unittest.mock import patch
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1];LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(LEGACY),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON
import driver as D
import parallel_events as parallel
HOST='172.29.189.119';UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'

def validate():
    for f,h in read(WORK/'GPU7_RESUME_PLAN.json')['source'].items():
        if sha(f)!=h:raise ValueError('Resume source changed')

def validate_gpu():
    if HOST not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong host')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise ValueError('Only GPU7 may be exposed')
    if subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=UUID:
        raise ValueError('Wrong GPU UUID')

def runtime_policy(policy):
    validate_gpu()
    # Memory placement is recorded separately from the frozen detector protocol.
    # All layers still compute on CUDA in the original bf16 precision.
    return dict(policy,physical_gpu=7,node11_gpu_uuid=UUID,gpu_memory_utilization=.36,cpu_offload_gb=28)

def adapted_infer():
    import p2_event_presence.inference as native
    source=inspect.getsource(native.infer)
    needle="policy = contract['policy']"
    if source.count(needle)!=1:raise ValueError('Native inference changed')
    source=source.replace(needle,needle+'; policy = runtime_policy(policy)')
    namespace=dict(native.__dict__,runtime_policy=runtime_policy)
    exec(compile(source,str(Path(__file__))+'::gpu7_memory_overlay','exec'),namespace)
    namespace['infer'].__adapted_source__=source
    return namespace['infer']

def persistent_factory(factory):
    cache={}
    def load(**kwargs):
        key=json.dumps(kwargs,sort_keys=True)
        if cache and key not in cache:raise ValueError('Detector model/runtime changed between groups')
        if key not in cache:cache[key]=factory(**kwargs)
        return cache[key]
    return load,cache

def local_cpu(events,cluster,stage):
    from p2_event_presence.contracts import rows
    executed=[]
    def local_launch(cl,commands,logs):
        executed.append(True)
        logs.mkdir(parents=True,exist_ok=False)
        def one(rank):
            out=events/f'cpu_{stage}_shards'/f'{rank:02d}'
            cfg=read(out/'contract.json')['policy'];py=cfg['media_python']if stage=='prepare'else str(PYTHON)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
            env['LD_LIBRARY_PATH']=str(Path(py).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            with(out/'gpu7_cpu_resume.log').open('ab',buffering=0)as log:
                subprocess.run([py,'-u',str(Path(parallel.__file__)),'cpu_worker','--cpu-stage',stage,'--work',str(out)],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ThreadPoolExecutor(4)as pool:list(pool.map(one,range(28)))
        save(logs/'completed.json',dict(returncodes=[0]*28,host=HOST,cpu_workers=4,all_cpu_shards=28))
    with patch('listen_cluster.launch',local_launch):parallel.parallel_cpu(events,cluster,stage)
    if executed:save(events/f'GPU7_CPU_{stage.upper()}_EXECUTION.json',dict(host=HOST,cpu_workers=4,logical_shards=28,all_cpu_on_node11=True))

def prepare_work(work):
    from p2_event_presence.contracts import rows,save_rows
    import p2_event_presence.media as media
    from reference_media_cache import bind
    p=read(work/'protocol.json');arm=p['arms'][0]['name'];events=work/'events'
    if not(events/'contract.json').exists():
        manifest=work/'event_pairs_manifest.jsonl'
        values=[r for i in range(28)for r in rows(work/'metric_manifests'/arm/'seamless'/f'{i:02d}.jsonl')]
        assert len(values)==64 and len({r['id']for r in values})==64
        save_rows(manifest,values)
        # Reuse the exact detector policy of the completed baseline (including its identity).
        policy=read(D.folder('baseline',137)/'events/contract.json')['policy'];cfg=work/'event_detector_config.json'
        save(cfg,policy,immutable=True);media.bind([manifest],events,cfg,method=arm)
    local_cpu(events,read(work/'cluster.json'),'prepare');bind(events,LEGACY/'reference_media_cache')
    local_cpu(events,read(work/'cluster.json'),'candidates')
    return events

def worker():
    validate();validate_gpu()
    from stage_driver import verify_scoring
    pending=[]
    for name in D.NAMES:
        for seed in D.SEEDS:
            work=D.folder(name,seed)
            if(work/'SCORING_COMPLETE.json').exists():verify_scoring(work);continue
            if not(work/'FINAL_AUDIT.json').exists():raise ValueError('Resume only finished generation/video metrics')
            pending.append(work)
    import vllm
    from p2_event_presence.cli import main
    from p2_event_presence.media import verify
    load,cache=persistent_factory(vllm.LLM)
    with patch.object(vllm,'LLM',load),concurrent.futures.ThreadPoolExecutor(1)as prep:
        future=prep.submit(prepare_work,pending[0])if pending else None
        for index,work in enumerate(pending):
            if(WORK/'STOP').exists():return
            save(WORK/'STATUS.json',dict(stage='gpu7_preparing_or_prefetching',group=work.name,at=time.time(),training_resumed=False))
            events=future.result()
            future=prep.submit(prepare_work,pending[index+1])if index+1<len(pending)else None
            save(WORK/'STATUS.json',dict(stage='gpu7_event_inference',group=work.name,at=time.time(),training_resumed=False))
            overlay=dict(host=HOST,physical_gpu=7,uuid=UUID,gpu_memory_utilization=.36,cpu_offload_gb=28,
                frozen_policy_identity=read(events/'contract.json')['policy_identity'],
                semantics='Execution/memory placement overlay only; unchanged bf16 model, native inputs/prompts/seed/schema',
                dispatcher_sha256=sha(Path(__file__)))
            save(events/'GPU7_RUNTIME_OVERLAY.json',overlay,immutable=True)
            adapted_infer()(events)
            expected=read(D.folder('baseline',137)/'events/detector_contract.json')['identity_hash']
            if read(events/'detector_contract.json')['identity_hash']!=expected:raise ValueError('Detector identity differs from baseline')
            from p2_event_presence.contracts import rows
            from p2_event_presence.scoring import score
            # Keep the native score stage, including explicit unknown/error states.
            args=[str(PYTHON),str(ROOT/'scripts/p2_event_presence.py'),'score','--work',str(events)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'))
            with(events/'gpu7_score.log').open('a')as log:
                result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode not in(0,2):raise RuntimeError('Native event scoring failed')
            summary=read(events/'summary.json')
            save(work/'EVENTS_COMPLETE.json',dict(summary_sha256=sha(events/'summary.json'),status=summary['status'],
                detector_based=True,human_calibrated=False,pilot_only=True,independent_of_training_event_head=True))
            save(events/'PARALLEL_EXECUTION_ACCEPTED.json',dict(success=True,workers=1,mode='single_gpu7_resume',
                runtime_overlay=overlay,runtime_overlay_sha256=sha(events/'GPU7_RUNTIME_OVERLAY.json')))
            D.seal_score(work)
    # The final comparison uses the same Python/numpy environment as before.
    subprocess.run([str(PYTHON),str(Path(__file__)),'report'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=512,training_remains_stopped=True,resume_device='node11GPU7'))
    save(WORK/'STATUS.json',dict(stage='awaiting_experiment_direction',at=time.time(),training_resumed=False))

def supervisor():
    validate();validate_gpu()
    import rolling_driver as cleanup
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit('Safe GPU7 cancellation')))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',VLLM_PORT='26500',MASTER_PORT='26530',VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    cfg=read(ROOT/'configs/p2_event_presence_v1.json');py=cfg['inference_python'];env['LD_LIBRARY_PATH']=str(Path(py).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    with(WORK/'gpu7_resume_worker.log').open('ab',buffering=0)as log:
        p=subprocess.Popen([py,'-u',str(Path(__file__)),'worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        job=dict(process=p,owned_pids={p.pid:cleanup.process_identity(p.pid)})
        try:
            while p.poll()is None:cleanup.remember_children(job);time.sleep(1)
            if p.returncode:raise RuntimeError('GPU7 resume worker failed; inspect gpu7_resume_worker.log')
        finally:cleanup.clean_worker(job)

def dispatch():
    validate()
    from listen_cluster import transport
    with(WORK/'rolling_controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[-1].split()[19]
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,dispatcher='node11GPU7',at=time.time()))
        cl=read(WORK/'PLAN.json')['cluster'];node=cl['nodes'][3]
        guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
        argv=[str(PYTHON),str(guard),'--guard','env','CUDA_VISIBLE_DEVICES=7','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',
              str(PYTHON),'-u',str(Path(__file__)),'supervisor']
        D.command(transport(cl,node,argv),WORK/'gpu7_resume_remote.log')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dispatch','supervisor','worker','report']);a=parser.parse_args()
    try:
        if a.action=='report':
            D.report();result=read(WORK/'COMPARISON.json');result['gpu7_resume_plan_sha256']=sha(WORK/'GPU7_RESUME_PLAN.json');save(WORK/'COMPARISON.json',result)
        else:globals()[a.action]()
    except Exception as exc:
        save(WORK/'GPU7_RESUME_FAILURE.json',dict(action=a.action,error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        if a.action=='dispatch':save(WORK/'STATUS.json',dict(stage='gpu7_resume_failed',at=time.time(),training_resumed=False))
        raise
