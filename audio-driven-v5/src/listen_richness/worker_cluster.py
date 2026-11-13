"""One resident TP=1 model per GPU, bounded multi-chunk submission and resumption."""
from functools import lru_cache
import hashlib,json,os,signal,subprocess,threading,time
from pathlib import Path
from .common import read,digest
from .extraction import output_schema
from .production import contract,messages,validation_error,statistic_record
from listen_benchmark.j_annotation import JobQueue,acquire,write


@lru_cache(maxsize=1024)
def schema_for_count(n):return output_schema(' '.join(['X.']*n))


def load_records(path,sha=None):
    raw=Path(path).read_bytes()
    if sha and hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Packed input changed')
    rs=[json.loads(line) for line in raw.splitlines() if line]
    if len({r['sample_id'] for r in rs})!=len(rs):raise ValueError('Duplicate requests')
    for r in rs:
        if digest(r['snapshot'])!=r['snapshot_sha256']:raise ValueError('Snapshot hash mismatch')
    return rs


def atomic_jsonl(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(llm,requests,profile,state,status,stop):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    tokenizer=llm.get_tokenizer();attempts=[[] for _ in requests];pending=list(range(len(requests)));rounds=[]
    for attempt in range(1,3):
        if not pending or stop.is_set():break
        prompts=[];sampling=[];ids=[]
        for i in pending:
            r=requests[i];tokens=r['prompt_token_ids']
            if attempt>1:
                m=messages(r['snapshot']['caption'])+[dict(role='user',content='Previous attempt failed validation: '+attempts[i][-1]['error']+'. Return a fresh extraction from the original caption.')]
                tokens=tokenizer.apply_chat_template(m,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if len(tokens)+8192>12288:continue  # preserve the first attempt; no truncation
            ids.append(i);prompts.append(dict(prompt_token_ids=tokens))
            sampling.append(SamplingParams(temperature=0,max_tokens=8192,
                structured_outputs=StructuredOutputsParams(json=schema_for_count(r['sentence_count']))))
        if not ids:break
        state.update(stage='generating',attempt=attempt,submitted=len(ids));status();start=time.monotonic()
        outputs=llm.generate(prompts,sampling,use_tqdm=False)
        if len(outputs)!=len(ids):raise RuntimeError('Inference output cardinality differs')
        retry=[];tokens_out=0
        for i,p,out in zip(ids,prompts,outputs):
            if out.prompt_token_ids!=p['prompt_token_ids']:raise ValueError('Output prompt identity differs')
            ans=out.outputs[0];r=requests[i]
            error=validation_error(ans.text,r['snapshot']['caption'],ans.finish_reason)
            a=dict(attempt=attempt,raw=ans.text,finish_reason=ans.finish_reason,completion_tokens=len(ans.token_ids),
                   prompt_tokens=len(out.prompt_token_ids),prompt_tokens_sha256=digest(out.prompt_token_ids),error=error)
            attempts[i].append(a);tokens_out+=len(ans.token_ids)
            if error:retry.append(i)
        rounds.append(dict(attempt=attempt,count=len(ids),seconds=time.monotonic()-start,output_tokens=tokens_out,strict_failed=len(retry)))
        pending=retry
    return attempts,rounds


def save_chunk(work,jid,records,attempts,profile,rounds,node,gpu):
    work=Path(work);raw_path=work/'attempts'/f'{jid}.jsonl';stat_path=work/'statistics'/f'{jid}.jsonl'
    raw_sha=atomic_jsonl(raw_path,[dict(sample_id=r['sample_id'],snapshot_sha256=r['snapshot_sha256'],attempts=a)
                                 for r,a in zip(records,attempts)])
    stats=[statistic_record(r,a,raw_path) for r,a in zip(records,attempts)]
    for i,s in enumerate(stats):s['extraction_record_index']=i
    stat_sha=atomic_jsonl(stat_path,stats)
    receipt=dict(job_id=jid,clips=len(records),input_ids_sha256=digest([r['sample_id'] for r in records]),
                 contract=contract(),profile=profile,node=node,gpu=gpu,completed_at=time.time(),
                 raw_file=str(raw_path),raw_sha256=raw_sha,statistics_file=str(stat_path),statistics_sha256=stat_sha,
                 strict_failed=sum(bool(a and a[-1].get('error')) for a in attempts),
                 incomplete_generations=sum(bool(a and a[-1].get('finish_reason')!='stop') for a in attempts),
                 unresolved=sum(s['record_status']=='unresolved_extraction' for s in stats),rounds=rounds,
                 rounds_scope='shared_reservoir_not_additive_across_chunks')
    write(work/'persisted'/f'{jid}.json',receipt)
    return receipt


def worker(work,node,gpu,profile,benchmark_input=None):
    work=Path(work);cfg=read(work/'config.json');name=f'{node}-gpu-{gpu}'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong visible GPU')
    expected=cfg['nodes'][node]['gpu_uuids'][str(gpu)]
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if raw[0].strip()!=expected or float(raw[1])>1024:raise RuntimeError('Assigned GPU identity differs or is occupied')
    owner=acquire(work/'worker_locks'/f'{name}.lock',False)
    if owner is None:raise RuntimeError('Worker already running')
    for p,h in cfg['code_hashes'].items():
        if hashlib.sha256((Path(cfg['project_root'])/p).read_bytes()).hexdigest()!=h:raise ValueError('Code changed: '+p)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    state=dict(pid=os.getpid(),node=node,gpu=gpu,profile=profile,started_at=time.time(),stage='loading_model',completed=0)
    def status():write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
    def heartbeat():
        while not stop.wait(10):status()
    status();threading.Thread(target=heartbeat,daemon=True).start()
    start=time.monotonic()
    try:
        from vllm import LLM
        sizes=[s for s in (1,2,4,8,16,32,64,128,256,512) if s<=profile['max_num_seqs']]
        llm=LLM(model=contract()['model'],dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=.92,max_model_len=12288,max_num_seqs=profile['max_num_seqs'],
            max_num_batched_tokens=profile['max_num_batched_tokens'],enable_chunked_prefill=True,
            enable_prefix_caching=True,enforce_eager=False,seed=20261031,
            compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=sizes[-1]),
            structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
        state.update(load_seconds=time.monotonic()-start,stage='ready');status()
        if benchmark_input:
            records=load_records(benchmark_input)
            t=time.monotonic();attempts,rounds=generate(llm,records,profile,state,status,stop)
            if stop.is_set():raise InterruptedError('Benchmark interrupted; not a valid timing')
            output=work/'benchmark'/name
            receipt=save_chunk(output,'benchmark',records,attempts,profile,rounds,node,gpu)
            elapsed=time.monotonic()-t
            state.update(stage='complete',completed=len(records),generation_seconds=sum(x['seconds'] for x in rounds),
                         processing_seconds=elapsed,clips_per_second=len(records)/elapsed,receipt=receipt)
            write(output/'summary.json',dict(state,finished_at=time.time()));status()
        else:
            q=JobQueue(work,'extract')
            while (work/'WAIT_START').exists() and not stop.is_set():stop.wait(.5)
            while not stop.is_set():
                if (work/'DRAIN').exists():break
                claims=[];jobs=[];records=[]
                try:
                    for _ in range(profile.get('reservoir_chunks',2)):
                        claim=q.claim(name)
                        if claim is None:break
                        jid,handle=claim;claims.append((jid,handle));job=read(work/'jobs'/f'{jid}.json')
                        if job['contract']!=contract():raise ValueError('Input prompt contract differs')
                        rs=load_records(job['path'],job['sha256'])
                        p=work/'persisted'/f'{jid}.json'
                        if p.exists():
                            receipt=read(p)
                            if receipt['input_ids_sha256']!=digest([r['sample_id'] for r in rs]) or receipt['contract']!=contract():raise ValueError('Persisted input differs')
                            for key in ('raw','statistics'):
                                if hashlib.sha256(Path(receipt[key+'_file']).read_bytes()).hexdigest()!=receipt[key+'_sha256']:raise ValueError('Persisted output changed')
                            q.finish(jid,receipt=receipt);handle.close();claims.pop();continue
                        jobs.append((jid,len(rs)));records.extend(rs)
                    if not records:
                        qs=read(q.path)
                        if (work/'producer_complete.json').exists() and not qs['inflight'] and qs['offset']>=(work/'ready.jsonl').stat().st_size:break
                        state['stage']='waiting';status();stop.wait(2);continue
                    state['jobs']=[j for j,_ in jobs]
                    attempts,rounds=generate(llm,records,profile,state,status,stop)
                    if any(not a for a in attempts) or (stop.is_set() and any(a[-1].get('error') and len(a)<2 for a in attempts)):
                        continue  # interruption cannot bypass the normal retry policy
                    offset=0
                    for jid,n in jobs:
                        receipt=save_chunk(work,jid,records[offset:offset+n],attempts[offset:offset+n],profile,rounds,node,gpu)
                        q.finish(jid,receipt=receipt);offset+=n;state['completed']+=n
                    state.update(stage='ready',last_progress_at=time.time());status()
                except Exception as exc:
                    state['last_error']=repr(exc);status()
                    for jid,_ in claims:q.finish(jid,error=repr(exc))
                    raise  # supervisor restarts with a fresh engine
                finally:
                    for _,handle in claims:handle.close()
            state['stage']='stopped' if stop.is_set() else 'complete';status()
    except BaseException as exc:
        state.update(stage='error',last_error=repr(exc));status();raise
    finally:stop.set();owner.close()
