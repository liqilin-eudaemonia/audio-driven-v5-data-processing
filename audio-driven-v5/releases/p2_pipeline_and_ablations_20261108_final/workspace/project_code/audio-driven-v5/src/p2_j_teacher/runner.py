"""Bounded CPU prefetch, batched GPU extraction, dynamic claims and final audit."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch

from p2_listen.contracts import read, atomic_json, sha, digest
from .core import Reader, load_model, extract_batch, persist
from .media import initialize_processor, prepare


def claim(work, jobs):
    for index, ids in enumerate(jobs):
        name=f'{index:06d}'
        if (work/'done'/(name+'.json')).exists() or (work/'failed'/(name+'.json')).exists():continue
        path=work/'locks'/(name+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
        lock=path.open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if (work/'done'/(name+'.json')).exists() or (work/'failed'/(name+'.json')).exists():lock.close();continue
        return name,ids,lock
    return None


def prepared_stream(reader,ids,pool,size,grid,prefetch):
    source=reader.source.manifest;queue=deque();iterator=iter(ids)
    def submit():
        sid=next(iterator,None)
        if sid is None:return
        item=reader.item(sid)
        if reader.load(sid,missing_ok=True) is not None:
            queue.append((item,None));return
        future=pool.submit(prepare,item,source.get('video_sampling_overrides',{}).get(sid),size,grid)
        queue.append((item,future))
    for _ in range(prefetch):submit()
    while queue:
        item,future=queue.popleft()
        value=None if future is None else future.result(timeout=180)
        submit();yield item,value


def worker(work,gpu,batch=16,producers=4,prefetch=32):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    reader=Reader(work);work=Path(work);jobs=read(work/'jobs.json')
    runtime,processor,size,grid=load_model(reader)
    pool=ProcessPoolExecutor(max_workers=producers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_processor,initargs=(reader.protocol['model'],))
    writes=ThreadPoolExecutor(max_workers=2);pending=deque();count=0;started=time.time()
    def state(status,**extra):
        atomic_json(work/'workers'/f'worker{gpu}.json',dict(status=status,worker=gpu,gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES',gpu)),pid=os.getpid(),time=time.time(),
             samples=count,elapsed_s=time.time()-started,batch=batch,producers=producers,prefetch=prefetch,**extra))
    state('ready')
    try:
        while True:
            job=claim(work,jobs)
            if job is None:break
            name,ids,lock=job;t=time.time();receipts=[];items=[];values=[];state('running',job=name)
            def flush():
                nonlocal count,items,values
                if not values:return
                out=extract_batch(runtime,values)
                for item,value in zip(items,out):pending.append(writes.submit(persist,reader,item,value))
                count+=len(values);items=[];values=[]
                while len(pending)>batch*2:receipts.append(pending.popleft().result())
                state('running',job=name)
            try:
                for item,value in prepared_stream(reader,ids,pool,size,grid,prefetch):
                    if value is None:
                        folder=work/'cache'/item['sample_id'];marker=read(folder/'complete.json')
                        receipts.append(dict(sample_id=item['sample_id'],bytes=(folder/'payload.pt').stat().st_size,
                                             payload_sha256=marker['payload_sha256'],identity=marker['identity']))
                        continue
                    items.append(item);values.append(value)
                    if len(values)>=batch:flush()
                flush()
                while pending:receipts.append(pending.popleft().result())
                if {r['sample_id'] for r in receipts}!=set(ids) or len(receipts)!=len(ids):raise ValueError('Incomplete task receipt')
                atomic_json(work/'done'/(name+'.json'),dict(success=True,samples=len(ids),sample_ids=ids,receipts=receipts,
                    seconds=time.time()-t,gpu=gpu,protocol_sha256=reader.protocol_sha))
            except Exception:
                atomic_json(work/'failed'/(name+'.json'),dict(success=False,error=traceback.format_exc(),sample_ids=ids,gpu=gpu,time=time.time()))
                raise
            finally:lock.close()
        state('finished')
    except BaseException:
        state('failed',error=traceback.format_exc());raise
    finally:
        # Avoid blocking forever on a native decoder if a future timed out.
        for process in list((getattr(pool,'_processes',None) or {}).values()):
            if process.is_alive():process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);writes.shutdown(wait=True)


def status(work):
    work=Path(work);p=read(work/'protocol.json');finished=list((work/'done').glob('*.json'))
    samples=0;seconds=0
    for path in finished:
        r=read(path);samples+=r['samples'];seconds+=r['seconds']
    return dict(time=time.time(),samples=samples,total=p['samples'],finished_jobs=len(finished),
                failures=[str(f) for f in (work/'failed').glob('*.json')],
                workers=[read(f) for f in (work/'workers').glob('*.json')])


def audit(work):
    reader=Reader(work);work=Path(work);jobs=read(work/'jobs.json');seen=set();total_bytes=0
    if list((work/'failed').glob('*.json')):raise ValueError('Extraction has unresolved failures')
    for index,ids in enumerate(jobs):
        receipt=read(work/'done'/f'{index:06d}.json')
        if not receipt['success'] or receipt['sample_ids']!=ids or receipt['protocol_sha256']!=reader.protocol_sha:
            raise ValueError('Wrong task completion receipt')
        if set(ids)&seen:raise ValueError('Duplicate completed members')
        seen.update(ids)
        if len(receipt['receipts'])!=len(ids) or {r['sample_id'] for r in receipt['receipts']}!=set(ids):raise ValueError('Invalid per-sample receipts')
        for r in receipt['receipts']:
            sid=r['sample_id'];folder=work/'cache'/sid;marker=read(folder/'complete.json')
            expected=reader.identity(reader.item(sid))
            if (marker['contract']!=expected or marker['identity']!=digest(expected)
                    or marker['payload_sha256']!=r['payload_sha256'] or marker['identity']!=r['identity']
                    or (folder/'payload.pt').stat().st_size!=r['bytes']):
                raise ValueError('Invalid teacher cache: '+sid)
            total_bytes+=r['bytes']
    if seen!=reader.ids:raise ValueError('Teacher membership differs')
    result=dict(success=True,samples=len(seen),payload_bytes=total_bytes,finished_at=time.time(),
                protocol_sha256=reader.protocol_sha,source=reader.protocol['source'],
                writer_validation='per-payload SHA256 and exact tensor roundtrip',
                final_validation='complete member coverage, source identity, per-sample receipt/hash binding and byte size')
    atomic_json(work/'COMPLETE.json',result,immutable=True);return result


def node(work,batch=8,producers=4,prefetch=16,per_gpu=2):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    work.joinpath('logs').mkdir(exist_ok=True)
    script=Path(__file__).resolve().parents[2]/'scripts/p2_j_teacher.py'
    if per_gpu not in (1,2):raise ValueError('Checked process budget is one or two per GPU')
    children={};restarts={i:0 for i in range(8*per_gpu)};files=[]
    def launch(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu//per_gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYAV_LOGGING='off',PYTHONUNBUFFERED='1',
                 PYTORCH_ALLOC_CONF='expandable_segments:True')
        log=open(work/'logs'/f'gpu{gpu}.log','a');files.append(log)
        args=[sys.executable,str(script),'worker','--work',str(work),'--gpu',str(gpu),
              '--batch',str(batch),'--producers',str(producers),'--prefetch',str(prefetch)]
        process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[gpu]=process
        atomic_json(work/'launchers'/f'gpu{gpu}.json',dict(pid=process.pid,command=args,time=time.time()))
    def stop(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    try:
        for gpu in range(8*per_gpu):launch(gpu)
        while True:
            active=False
            for gpu,process in list(children.items()):
                code=process.poll()
                if code is None:active=True;continue
                if code!=0 and restarts[gpu]<2:
                    # Owned process group only; terminate any orphan decoder children.
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    restarts[gpu]+=1;launch(gpu);active=True
            current=status(work);atomic_json(work/'status.json',current)
            with open(work/'status_history.jsonl','a') as f:f.write(json.dumps(current)+'\n')
            output=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw',
                                   '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
            with open(work/'resources.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=output.stdout))+'\n')
            if not active:break
            time.sleep(20)
        return audit(work)
    finally:
        for process in children.values():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for f in files:f.close()
