#!/usr/bin/env python3
"""Native Gemma text-only cache for the frozen current train+dev 500h.

Reuse NativeTextConditioner.encode_batch, FixedPrompt, long-text encoding,
checked atomic tensor persistence, current H row reader, and shared claims.
Never start a VAE, Qwen forward, teacher, Whisper, labeling, or training.
"""
import argparse
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from functools import lru_cache
import fcntl
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import activate,extractor_config
from p3_listen.prompts import GEMMA,FixedPrompt,contract as prompts
from p2_j_cache.data import caption_prompt
from p2_j_current_qwen_hidden import Reader as HReader,ensure,claim,env,remote,probe
DEFAULT_WORK=ROOT/'experiments/p2_j_train_dev500h_gemma_cache_20261105'
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'


def code_identity():
    files=[Path(__file__),ROOT/'scripts/p2_j_current_qwen_hidden.py',ROOT/'src/p2_j_cache/runtime.py',
           ROOT/'src/p2_j_cache/data.py',ROOT/'src/p2_plan_cache/encoding.py',ROOT/'src/p3_listen/prompts.py']
    return {str(p):sha(p) for p in files}


def prepare(work):
    from p2_plan_cache.runtime import dependencies
    source=HReader(SOURCE);complete=read(SOURCE/'COMPLETE.json');ensure(complete['success'] and complete['samples']==284218,'Incomplete current Qwen H')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    p=dict(schema='current-j-train-dev-gemma-native-v1',source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),
           source_complete_sha256=sha(SOURCE/'COMPLETE.json'),selection_sha256=source.protocol['selection_sha256'],
           sample_ids=source.protocol['sample_ids'],samples=source.protocol['samples'],source_split_counts=source.protocol['source_split_counts'],
           canary_ids=source.protocol['canary_ids'],config=source.config,dependencies=dependencies(source.config,'text'),code=code_identity(),
           prompts=prompts(),positive_fixed=GEMMA,negative_fixed='',guidance='same exact full caption prompt as completed Qwen input_plan',
           parts=['text','text_guided'],batch_size=32,native_sequence_length=1024,max_text_tokens=8192,truncate=False,
           writer_policy='atomic payload; SHA256 and exact tensor readback',only_text=True)
    atomic_json(work/'protocol.json',p,immutable=True)
    ids=p['sample_ids'];atomic_json(work/'jobs.json',[ids[i:i+256] for i in range(0,len(ids),256)],immutable=True)
    atomic_json(work/'cluster.json',read(SOURCE/'cluster.json'),immutable=True)
    print(json.dumps(dict(prepared=True,samples=p['samples'],source_split_counts=p['source_split_counts'])),flush=True)


class Reader:
    def __init__(self,work,verify=True):
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol;self.protocol_sha=sha(self.work/'protocol.json')
        ensure(p['code']==code_identity() and p['prompts']==prompts(),'Gemma code/prompt changed; create a new revision')
        ensure(sha(Path(p['source'])/'protocol.json')==p['source_protocol_sha256'] and sha(Path(p['source'])/'COMPLETE.json')==p['source_complete_sha256'],'Completed Qwen source changed')
        self.source=HReader(p['source'],verify=False);self.config=p['config'];activate(self.config)
        if verify:
            from p2_plan_cache.runtime import dependencies
            ensure(dependencies(self.config,'text')==p['dependencies'],'Gemma/LTX text assets changed')
    def identity(self,row=None):
        result=dict(schema=self.protocol['schema'],protocol_sha256=self.protocol_sha,part='text' if row is None else 'text_guided')
        if row is not None:result.update(sample_id=row['sample_id'],source_split=row['split'],caption_sha256=row['caption_file_sha256'],prompt_sha256=digest(caption_prompt(row['caption'])))
        return result
    def row(self,sid):return self.source.row(sid)
    def fixed(self,device='cpu'):
        from p2_listen.cache import load_part
        return load_part(self.work/'cache/text/shared',self.identity(),device)
    def load(self,sid,device='cpu',missing_ok=False):
        from p2_listen.cache import load_part
        row=self.row(sid);v=load_part(self.work/'cache/text_guided'/sid,self.identity(row),device,missing_ok)
        if v is not None:validate(row,v)
        return v


def validate(row,value):
    from p2_plan_cache.runtime import validate_payload
    p=caption_prompt(row['caption']);validate_payload(value,dict(guidance=dict(prompt=p,prompt_sha256=digest(p))),'text')


def checked_prompt(reader,row):
    p=caption_prompt(row['caption']);folder=Path(reader.protocol['source'])/'cache/input_plan'/row['sample_id']
    marker=read(folder/'complete.json');expected=reader.source.identity(row,'input_plan')
    ensure(marker['contract']==expected and marker['identity']==digest(expected),'Qwen guidance caption/source binding differs')
    return p


def load_session(reader):
    from omni_av.ltx_runtime import NativeTextConditioner
    return NativeTextConditioner(extractor_config(reader.config),'cuda:0')


def encode_batch(session,rows,reader):
    from p2_plan_cache.encoding import encode_text
    from p2_listen.cache import cpu
    ps=[checked_prompt(reader,row) for row in rows]
    tokenizer=session.encoder.tokenizer
    counts=[len(tokenizer.tokenizer(p.strip(),truncation=False)['input_ids']) for p in ps]
    if max(counts)<=1024:
        # Assert exact native input IDs and masks, including all nonpadding tokens.
        for p,n in zip(ps,counts):
            pairs=tokenizer.tokenize_with_weights(p)['gemma'];ids=tokenizer.tokenizer(p.strip(),truncation=False)['input_ids']
            ensure([int(t) for t,valid in pairs if int(valid)]==ids and len(pairs)==1024 and n==len(ids),'Native Gemma tokenizer changed/truncated guidance')
        values=session.encode_batch([FixedPrompt(p) for p in ps])
    else:values=[encode_text(session,p,8192)[0] for p in ps]
    result=[]
    for row,p,n,value in zip(rows,ps,counts,values):
        value=dict(positive=cpu(value),prompt_sha256=digest(p),tokens=n,sequence_length=value['mask'].shape[1],truncated=False)
        validate(row,value);result.append(value)
    return result


def write_value(reader,row,value):
    from p2_j_cache.runtime import persist
    sid=row['sample_id'];persist(reader.work,'text_guided',sid,reader.identity(row),value)
    folder=reader.work/'cache/text_guided'/sid;m=read(folder/'complete.json')
    return dict(sample_id=sid,split=row['split'],caption_sha256=row['caption_file_sha256'],prompt_sha256=value['prompt_sha256'],
                identity=m['identity'],payload_sha256=m['payload_sha256'],bytes=(folder/'payload.pt').stat().st_size)


def canary(work):
    import torch
    from p2_listen.cache import cpu
    from p2_j_cache.runtime import persist
    r=Reader(work);torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();s=load_session(r)
    fixed=dict(zip(('positive','negative'),map(cpu,s.encode_batch([FixedPrompt(GEMMA),FixedPrompt('')]))))
    persist(r.work,'text','shared',r.identity(),fixed)
    rows=[r.row(sid) for sid in r.protocol['canary_ids']];values=encode_batch(s,rows,r)
    # Same batch shape and exact native code, without orchestration/payload wrappers.
    with torch.no_grad():baseline=s.encode_batch([FixedPrompt(caption_prompt(row['caption'])) for row in rows])
    comparisons=[]
    for row,v,ref in zip(rows,values,baseline):
        for k in ('video','audio','mask'):ensure(torch.equal(v['positive'][k],ref[k].cpu()),'Batched Gemma differs from native same-batch reference')
        old=r.source.load(row['sample_id'],'input_plan');ensure(old['prompt']==v['positive']['prompt'] and old['prompt_sha256']==v['prompt_sha256'],'Qwen/Gemma full prompt differs')
        receipt=write_value(r,row,v);r.load(row['sample_id'])
        comparisons.append(dict(sample_id=row['sample_id'],split=row['split'],native_same_batch_exact=True,qwen_full_prompt_exact=True,tokens=v['tokens'],shape=list(v['positive']['video'].shape)))
    result=dict(success=True,protocol_sha256=r.protocol_sha,count=len(rows),rows=comparisons,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,time=time.time())
    atomic_json(r.work/'CANARY.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def worker(work,wid):
    import torch
    r=Reader(work);proof=read(r.work/'CANARY.json');ensure(proof['success'] and proof['protocol_sha256']==r.protocol_sha,'Matching Gemma canary required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();jobs=read(r.work/'jobs.json');session=None;count=0;start=time.time()
    def state(status,**kw):atomic_json(r.work/'workers'/(wid+'.json'),dict(worker=wid,status=status,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),samples=count,time=time.time(),started_at=start,**kw))
    state('starting')
    with ThreadPoolExecutor(max_workers=2) as writers:
        while True:
            task=claim(r.work,jobs)
            if task is None:break
            name,ids,lock=task;pending=deque();receipts=[];tick=time.time()
            try:
                rows=[r.row(sid) for sid in ids];missing=[]
                for row in rows:
                    cached=r.load(row['sample_id'],missing_ok=True)
                    if cached is None:missing.append(row)
                    else:receipts.append(write_value(r,row,cached))
                if missing and session is None:state('loading_model',job=name);session=load_session(r)
                for start_batch in range(0,len(missing),32):
                    batch=missing[start_batch:start_batch+32];state('encoding',job=name,batch=len(batch))
                    values=encode_batch(session,batch,r)
                    for row,v in zip(batch,values):pending.append(writers.submit(write_value,r,row,v))
                    count+=len(batch)
                    while len(pending)>=64:receipts.append(pending.popleft().result())
                    state('writing',job=name,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    del values,v
                while pending:receipts.append(pending.popleft().result())
                ensure(len(receipts)==len(ids) and {x['sample_id'] for x in receipts}==set(ids),'Incomplete Gemma block')
                atomic_json(r.work/'done'/(name+'.json'),dict(success=True,sample_ids=ids,samples=len(ids),receipts=receipts,
                            worker=wid,seconds=time.time()-tick,protocol_sha256=r.protocol_sha))
                state('running',job=name)
            except Exception:
                while pending:
                    try:pending.popleft().result()
                    except Exception:pass
                atomic_json(r.work/'failed'/(name+'.json'),dict(error=traceback.format_exc(),sample_ids=ids,worker=wid,time=time.time()))
                state('block_failed',job=name,error=traceback.format_exc())
            finally:lock.close()
    state('finished')


def status(work):
    work=Path(work);p=read(work/'protocol.json');rows=[read(f) for f in (work/'done').glob('*.json')]
    return dict(time=time.time(),samples=sum(x['samples'] for x in rows),total=p['samples'],finished_jobs=len(rows),
                failures=[str(f) for f in (work/'failed').glob('*.json')],workers=[read(f) for f in (work/'workers').glob('*.json')])


def node(work,name):
    folder=Path(work)/'nodes'/name;folder.mkdir(parents=True,exist_ok=True);children={};logs=[]
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for gpu in range(8):
                wid=f'{name}-g{gpu}';cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',work,'--worker-id',wid]
                log=(folder/(wid+'.log')).open('a');logs.append(log)
                children[wid]=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(folder/(wid+'.pid.json'),dict(pid=children[wid].pid,gpu=gpu))
            while True:
                states={k:v.poll() for k,v in children.items()};atomic_json(folder/'status.json',dict(time=time.time(),states=states,node=name,pid=os.getpid()))
                snap=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=snap.stdout))+'\n')
                if all(v is not None for v in states.values()):break
                time.sleep(10)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
        finally:
            for child in children.values():
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for f in logs:f.close()


@lru_cache(maxsize=1)
def audit_reader(work):return Reader(work,verify=False)


def audit_block(args):
    work,i,ids=args;r=audit_reader(work);d=read(r.work/'done'/f'{i:06d}.json');ensure(d['success'] and d['sample_ids']==ids and d['protocol_sha256']==r.protocol_sha,'Wrong completed text block')
    ensure(len(d['receipts'])==len(ids) and {x['sample_id'] for x in d['receipts']}==set(ids),'Text receipt membership differs')
    counts=Counter();total=0
    for v in d['receipts']:
        row=r.row(v['sample_id']);expected=r.identity(row);folder=r.work/'cache/text_guided'/row['sample_id'];m=read(folder/'complete.json')
        ensure(m['contract']==expected and m['identity']==digest(expected)==v['identity'],'Text cache identity mismatch')
        ensure(v['prompt_sha256']==digest(checked_prompt(r,row)) and v['caption_sha256']==row['caption_file_sha256'],'Text prompt or Qwen source mismatch')
        ensure(m['payload_sha256']==v['payload_sha256'] and (folder/'payload.pt').stat().st_size==v['bytes'],'Text payload receipt mismatch')
        counts[row['split']]+=1;total+=v['bytes']
    return dict(samples=len(ids),counts=dict(counts),bytes=total)


def audit(work):
    r=Reader(work);ensure(not list((r.work/'failed').glob('*.json')),'Unresolved text block failure');r.fixed()
    jobs=read(r.work/'jobs.json');ids=[s for block in jobs for s in block];ensure(ids==r.protocol['sample_ids'] and len(set(ids))==r.protocol['samples'],'Wrong full text membership')
    counts=Counter();total=0;samples=0;start=time.time()
    with ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for x in pool.map(audit_block,[(work,i,ids) for i,ids in enumerate(jobs)],chunksize=1):samples+=x['samples'];counts.update(x['counts']);total+=x['bytes']
    ensure(samples==r.protocol['samples'] and dict(counts)==r.protocol['source_split_counts'],'Wrong final split coverage')
    result=dict(success=True,samples=samples,source_split_counts=dict(counts),payload_bytes=total,parts=['text','text_guided'],fixed_text_complete=True,
                protocol_sha256=r.protocol_sha,source_qwen_protocol_sha256=r.protocol['source_protocol_sha256'],selection_sha256=r.protocol['selection_sha256'],
                qwen_gemma_full_prompt_bindings_verified=True,parallel_audit_seconds=time.time()-start,time=time.time(),vae_started=False,training_started=False)
    atomic_json(r.work/'COMPLETE.json',result,immutable=True);return result


def launch_remote(cluster,n,work):
    code='''import subprocess,sys,json,os,time
from pathlib import Path
root,work,name,script=sys.argv[1:];f=Path(work)/'nodes'/name;f.mkdir(parents=True,exist_ok=True);pf=f/'launch.json'
if pf.exists():
 d=json.loads(pf.read_text());p=Path('/proc')/str(d['pid'])/'cmdline'
 if p.exists() and script.encode() in p.read_bytes() and work.encode() in p.read_bytes():print(json.dumps(dict(already_running=True,pid=d['pid'])));sys.exit()
cmd=[sys.executable,script,'node','--work',work,'--node',name]
e=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
with (f/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,command=cmd,time=time.time());pf.write_text(json.dumps(d));print(json.dumps(d))'''
    return json.loads(remote(cluster,n,[cluster['python'],'-c',code,str(ROOT),str(work),n['name'],str(Path(__file__).resolve())]))


def chain(work):
    work=Path(work);r=Reader(work);cluster=read(work/'cluster.json')
    def state(stage,**kw):atomic_json(work/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (work/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (work/'COMPLETE.json').exists():state('complete');return
            if not (work/'production_started.json').exists():
                stable=0
                while stable<2:
                    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(lambda n:probe(cluster,n),cluster['nodes']))
                    available=all(x['available'] and all(g[1]>=138*1024 for g in x['gpu']) for x in checks)
                    stable=stable+1 if available else 0;state('resource_check',checks=checks,consecutive_clear=stable)
                    if stable<2:time.sleep(10)
                if not (work/'CANARY.json').exists():
                    state('gpu_canary')
                    with (work/'canary.log').open('a') as log:
                        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'canary','--work',str(work)],cwd=ROOT,env=env(0),stdout=log,stderr=subprocess.STDOUT)
                        ensure(child.wait()==0,'Gemma canary failed; see canary.log')
                proof=read(work/'CANARY.json');ensure(proof['success'] and proof['protocol_sha256']==r.protocol_sha,'Wrong Gemma canary')
                atomic_json(work/'execution.json',dict(model_workers=32,per_gpu=1,batch_size=32,job_size=256,writer_threads=2,pending_writes_max=64,native_text_encoder=True),immutable=True)
                atomic_json(work/'production_started.json',dict(time=time.time(),protocol_sha256=r.protocol_sha),immutable=True)
            with ThreadPoolExecutor(max_workers=4) as pool:launches=list(pool.map(lambda n:launch_remote(cluster,n,work),cluster['nodes']))
            state('extracting',launches=launches);last=-1;last_change=time.monotonic()
            while True:
                s=status(work);atomic_json(work/'cluster_status.json',s)
                with (work/'history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in s.items() if k!='workers'})+'\n')
                state('extracting',samples=s['samples'],total=s['total'],failures=s['failures'])
                if s['samples']==s['total']:state('auditing');result=audit(str(work));state('complete',result=result);return
                if s['samples']!=last:last=s['samples'];last_change=time.monotonic()
                if all((work/'nodes'/n['name']/'EXIT.json').exists() for n in cluster['nodes']):raise RuntimeError('Nodes exited before complete Gemma coverage')
                if time.monotonic()-last_change>1800:raise RuntimeError('No text block completion in 30 minutes')
                time.sleep(20)
        except BaseException:state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','canary','worker','node','chain','status','audit']);p.add_argument('--work',default=str(DEFAULT_WORK));p.add_argument('--worker-id',default='local');p.add_argument('--node',default='node12');a=p.parse_args()
    if a.command=='prepare':prepare(a.work)
    elif a.command=='canary':canary(a.work)
    elif a.command=='worker':worker(a.work,a.worker_id)
    elif a.command=='node':node(a.work,a.node)
    elif a.command=='chain':chain(a.work)
    elif a.command=='audit':print(json.dumps(audit(a.work)))
    else:s=status(a.work);s.pop('workers');print(json.dumps(s))

if __name__=='__main__':main()
