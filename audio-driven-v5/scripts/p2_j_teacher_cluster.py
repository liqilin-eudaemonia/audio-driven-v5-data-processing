#!/usr/bin/env python3
"""Multi-node orchestration and timing around the unchanged teacher functions."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
_fast_checked=None


def prepared_stream_ready(reader,ids,pool,size,grid,prefetch):
    """Same per-item preparation, bounded futures, no slow-item head blocking."""
    from concurrent.futures import wait,FIRST_COMPLETED
    source=reader.source.manifest;iterator=iter(ids);pending={};cached=deque();exhausted=False
    def fill():
        nonlocal exhausted
        while not exhausted and len(pending)+len(cached)<prefetch:
            sid=next(iterator,None)
            if sid is None:
                exhausted=True;break
            item=reader.item(sid)
            if reader.load(sid,missing_ok=True) is not None:
                cached.append(item);continue
            future=pool.submit(timed_prepare,item,source.get('video_sampling_overrides',{}).get(sid),size,grid)
            pending[future]=(item,time.monotonic())
    fill()
    while pending or cached:
        if cached:
            item=cached.popleft();fill();yield item,None;continue
        now=time.monotonic()
        overdue=[f for f,(_,t) in pending.items() if now-t>=180 and not f.done()]
        if overdue:raise TimeoutError('Teacher preparation exceeded 180 seconds: '+pending[overdue[0]][0]['sample_id'])
        ready,_=wait(pending,timeout=max(.001,min(180-(now-t) for _,t in pending.values())),return_when=FIRST_COMPLETED)
        for future in ready:
            item,_=pending.pop(future);value=future.result()
            fill();yield item,value


def timed_prepare(*args,**kwargs):
    global _fast_checked
    from p2_j_teacher import media
    times={};original_frames=media.selected_frames;original_audio=media.listener_wave
    base_frames=original_frames;execution=None
    if os.environ.get('J_TEACHER_FAST_DECODER'):
        import hashlib
        from p2_j_teacher_accel import selected_frames
        execution=json.loads(os.environ['J_TEACHER_FAST_DECODER'])
        if _fast_checked != execution:
            for path_key,hash_key in (('decoder_file','decoder_sha256'),('proof_file','proof_sha256')):
                if hashlib.sha256(Path(execution[path_key]).read_bytes()).hexdigest()!=execution[hash_key]:
                    raise ValueError('Unverified accelerated decoder changed')
            _fast_checked=execution
        def accelerated(*a,**k):return selected_frames(*a,**k,threads=execution['threads'])
        original_frames=accelerated
    def frames(*a,**k):
        t=time.monotonic();v=original_frames(*a,**k);times['frames_s']=time.monotonic()-t;return v
    def audio(*a,**k):
        t=time.monotonic();v=original_audio(*a,**k);times['wave_s']=time.monotonic()-t;return v
    media.selected_frames=frames;media.listener_wave=audio;t=time.monotonic()
    try:
        value=media.prepare(*args,**kwargs);times['prepare_s']=time.monotonic()-t
        if execution is not None:value['audit']['execution']=execution
        times['duration_s']=value['audit']['duration_s']
        value['_timing']=times;return value
    finally:media.selected_frames=base_frames;media.listener_wave=original_audio


def run_worker(a):
    from p2_j_teacher import runner
    runner.prepare=timed_prepare
    if a.completion_order:runner.prepared_stream=prepared_stream_ready
    original=runner.extract_batch;last=None
    path=Path(a.work)/'timings'/f'{a.node}-worker{a.worker_id}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    def extract(runtime,prepared):
        nonlocal last
        start=time.monotonic();value=original(runtime,prepared);end=time.monotonic()
        row=dict(time=time.time(),node=a.node,worker=a.worker_id,samples=len(prepared),
                 gpu_stage_s=end-start,between_gpu_calls_s=None if last is None else start-last,
                 prepare=[p.get('_timing') for p in prepared])
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
        last=end;return value
    runner.extract_batch=extract
    return runner.worker(a.work,a.worker_id,a.batch,a.producers,a.prefetch)


def run_node(a):
    from p2_listen.contracts import atomic_json,read
    work=Path(a.work);folder=work/'nodes'/a.node;folder.mkdir(parents=True,exist_ok=True)
    children={};attempts={i:0 for i in range(8*a.per_gpu)};logs=[]
    def launch(slot):
        wid=a.offset+slot;gpu=slot//a.per_gpu
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYAV_LOGGING='off',PYTHONUNBUFFERED='1',
                 PYTORCH_ALLOC_CONF='expandable_segments:True')
        if a.fast_decoder:
            env['J_TEACHER_FAST_DECODER']=Path(a.fast_decoder).read_text()
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',str(work),'--node',a.node,
             '--worker-id',str(wid),'--batch',str(a.batch),'--producers',str(a.producers),'--prefetch',str(a.prefetch)]
        if a.completion_order:cmd.append('--completion-order')
        log=open(folder/f'worker{wid}.log','a');logs.append(log)
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        children[slot]=child;atomic_json(folder/f'worker{wid}.pid.json',dict(pid=child.pid,gpu=gpu,worker=wid,command=cmd,time=time.time()))
    def stop(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for slot in attempts:launch(slot)
            while True:
                for slot,p in list(children.items()):
                    if p.poll() not in (None,0) and attempts[slot]<2:
                        try:os.killpg(p.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        attempts[slot]+=1;launch(slot)
                states={a.offset+s:p.poll() for s,p in children.items()}
                atomic_json(folder/'status.json',dict(time=time.time(),node=a.node,states=states,per_gpu=a.per_gpu,
                            batch=a.batch,producers=a.producers,prefetch=a.prefetch,completion_order=a.completion_order))
                output=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
                                       '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=output.stdout))+'\n')
                if all(x is not None for x in states.values()):break
                time.sleep(15)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),worker_exit_codes=states))
        finally:
            for p in children.values():
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for f in logs:f.close()


def watch(a):
    from p2_j_teacher.runner import status,audit
    from p2_listen.contracts import atomic_json,read
    work=Path(a.work)
    while True:
        s=status(work);atomic_json(work/'cluster_status.json',s)
        with (work/'cluster_history.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        if s['samples']==s['total'] and not s['failures']:
            if not (work/'COMPLETE.json').exists():return audit(work)
            return read(work/'COMPLETE.json')
        time.sleep(30)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['node','worker','watch'])
    p.add_argument('--work',required=True);p.add_argument('--node',default='node8')
    p.add_argument('--offset',type=int,default=0);p.add_argument('--worker-id',type=int,default=0)
    p.add_argument('--per-gpu',type=int,default=2);p.add_argument('--producers',type=int,default=4)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--prefetch',type=int,default=16)
    p.add_argument('--fast-decoder')
    p.add_argument('--completion-order',action='store_true')
    a=p.parse_args()
    if a.command=='worker':return run_worker(a)
    if a.command=='node':return run_node(a)
    return watch(a)

if __name__=='__main__':main()
