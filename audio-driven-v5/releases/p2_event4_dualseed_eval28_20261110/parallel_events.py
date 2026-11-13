"""Shard native independent media detection over the released28 GPUs."""
from pathlib import Path
import argparse,copy,inspect,json,os,subprocess,sys,time
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
LEGACY=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(LEGACY),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save

def assignment(cluster,rank):
    if not 0<=rank<28:raise ValueError('Invalid rank')
    node=cluster['nodes'][rank//8];gpu=rank%8
    if gpu not in node['gpus']:raise ValueError('Outside allocation')
    return node,gpu

def shards(media,n=28):
    by_id={m['media_id']:m for m in media}
    if any(by_id[m['media_id']]!=m for m in media):raise ValueError('Conflicting media identity')
    result=[[]for _ in range(n)]
    for i,key in enumerate(sorted(by_id)):result[i%n].append(by_id[key])
    return result

def adapt_infer():
    import p2_event_presence.inference as native
    source=inspect.getsource(native.infer)
    old="if os.environ.get('CUDA_VISIBLE_DEVICES') != str(policy['physical_gpu']): raise ValueError('Bind only authorized physicalGPU7')"
    if source.count(old)!=1:raise ValueError('Native inference device guard changed')
    source=source.replace(old,"execution=read(work/'EXECUTION.json'); validate_execution(execution); physical=execution['physical_gpu']")
    old="['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total'"
    if source.count(old)!=1:raise ValueError('Native query changed')
    source=source.replace(old,"['nvidia-smi','-i',str(physical),'--query-gpu=uuid,memory.free,memory.total'")
    source=source.replace("gpu[0].strip() != policy['node11_gpu_uuid']","gpu[0].strip() != execution['uuid']")
    namespace=dict(native.__dict__,validate_execution=validate_execution)
    exec(compile(source,str(Path(__file__))+'::native_infer','exec'),namespace)
    namespace['infer'].__adapted_source__=source
    return namespace['infer']

def validate_execution(execution):
    plan=read(WORK/'PLAN.json')
    for f,h in plan['source'].items():
        if sha(f)!=h:raise ValueError('Reviewed execution source changed')
    node,gpu=assignment(plan['cluster'],execution['rank'])
    if execution['host']!=node['host']or execution['physical_gpu']!=gpu:
        raise ValueError('Execution assignment changed')
    if node['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong detector host')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong detector GPU mask')
    actual=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual!=execution['uuid']:raise ValueError('Wrong detector GPU UUID')

def parallel_infer(events,cluster):
    from p2_event_presence.contracts import rows,save_rows
    from listen_cluster import transport,launch
    original=rows(events/'prepared_pairs.jsonl');media=[m for row in original for m in row['prepared_media'].values()]
    parts=shards(media);contract=read(events/'contract.json');prepared=read(events/'PREPARED.json')
    from p2_combined_whisper_eval import link
    for rank,values in enumerate(parts):
        out=events/'inference_shards'/f'{rank:02d}';out.mkdir(parents=True,exist_ok=True)
        save(out/'contract.json',contract,immutable=True);link(events/'frozen_pairs.jsonl',out/'frozen_pairs.jsonl')
        save_rows(out/'prepared_pairs.jsonl',[dict(prepared_media={'anonymous':m})for m in values])
        p=copy.deepcopy(prepared);p['prepared_pairs_sha256']=sha(out/'prepared_pairs.jsonl');save(out/'PREPARED.json',p)
    commands=[]
    for index,node in enumerate(cluster['nodes']):
        commands.append([str(cluster['python']),'-u',str(Path(__file__)),'node','--work',str(events),'--index',str(index)])
    launch(cluster,commands,events/f'parallel_launch_{time.time_ns()}')
    identities=[];detections={}
    for rank in range(28):
        out=events/'inference_shards'/f'{rank:02d}';d=read(out/'DETECTION_COMPLETE.json');identities.append(read(out/'detector_contract.json'))
        for key,value in d['detections'].items():
            if key in detections:raise ValueError('Duplicate detection task')
            detections[key]=value
    if len({x['identity_hash']for x in identities})!=1:raise ValueError('Detector identity differs across GPUs')
    expected={m['media_id']for m in media}
    if set(detections)!=expected:raise ValueError('Missing media detection')
    save(events/'detector_contract.json',identities[0])
    save(events/'DETECTION_COMPLETE.json',dict(media=len(expected),detector_identity=identities[0]['identity_hash'],
         detections=detections,completed_at=time.time(),pilot_only=True))
    save(events/'PARALLEL_EXECUTION_ACCEPTED.json',dict(success=True,workers=28,media=len(expected),
         same_native_prompts_and_sampling=True,execution_policy='Native canonical detector identity; actual per-worker device recorded separately',
         assignments=[read(events/'inference_shards'/f'{r:02d}'/'EXECUTION.json')for r in range(28)]))

def stage_adapter():
    import run_events as native
    source=inspect.getsource(native.run)
    old="if v.returncode not in allowed:raise RuntimeError('Event evaluation failed: '+str(log))"
    # Retain native bind/read/cue/scoring/cache code; only inference is dispatched.
    point="python=python or sys.executable;runtime_env="
    if source.count(point)!=1:raise ValueError('Native event stage changed')
    source=source.replace(point,"\n  if stage=='infer':\n   parallel_infer(events,cl)\n   return\n  python=python or sys.executable;runtime_env=")
    old_cfg="atomic_json(work/'event_detector_config.json',cfg,immutable=True)"
    if source.count(old_cfg)!=1:raise ValueError('Native detector configuration changed')
    source=source.replace(old_cfg,"cfg.update(gpu_memory_utilization=.80,cpu_offload_gb=0,authorized_scope='released28 GPUs; actual device bindings in inference_shards/EXECUTION.json; GPU7 UUID is the canonical shared detector-policy identifier')\n "+old_cfg)
    namespace=dict(native.__dict__,parallel_infer=parallel_infer)
    exec(compile(source,str(Path(__file__))+'::native_event_stages','exec'),namespace)
    return namespace['run']

def run(work):
    from p2_loss_search28.resources import idle
    if not idle(read(work/'cluster.json'))['idle']:raise ValueError('Detection requires released28-GPU allocation')
    stage_adapter()(work)

def node(events,index):
    cluster=read(WORK/'PLAN.json')['cluster'];n=cluster['nodes'][index];children=[];streams=[]
    if n['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong node')
    try:
        for gpu in n['gpus']:
            rank=index*8+gpu;out=events/'inference_shards'/f'{rank:02d}'
            uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            save(out/'EXECUTION.json',dict(host=n['host'],physical_gpu=gpu,uuid=uuid,rank=rank,source_sha256=sha(Path(__file__))),immutable=True)
            cfg=read(events/'contract.json')['policy'];py=cfg['inference_python'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1')
            env['LD_LIBRARY_PATH']=str(Path(py).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            stream=(out/'inference.log').open('ab',buffering=0);streams.append(stream)
            children.append(subprocess.Popen([py,'-u',str(Path(__file__)),'infer','--work',str(out)],env=env,stdout=stream,stderr=subprocess.STDOUT))
        while any(p.poll()is None for p in children):
            if any(p.poll()not in(None,0)for p in children):raise RuntimeError('Detector worker failed; inspect inference_shards logs')
            time.sleep(2)
        if any(p.returncode for p in children):raise RuntimeError('Detector worker failure')
    finally:
        for p in children:
            if p.poll()is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','node','infer']);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--index',type=int)
    a=parser.parse_args()
    if a.action=='run':run(a.work.resolve())
    elif a.action=='node':node(a.work.resolve(),a.index)
    else:adapt_infer()(a.work.resolve())
