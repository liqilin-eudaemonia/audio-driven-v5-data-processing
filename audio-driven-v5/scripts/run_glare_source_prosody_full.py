#!/usr/bin/env python3
"""Execution-only full prosody runner: bounded source prefetch and packed native batches."""
import argparse
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import queue
import signal
import sqlite3
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import ChunkCache,CHUNK_SAMPLES,IntegrityError,lock,save,read,sha,canonical,audit_raw
from glare_prosody_source_runtime import OfficialRuntime,waveform_sha
from prepare_glare_prosody_subset import BASE,OLD,protocol,connect,require


class PackedRuntime(OfficialRuntime):
    """Execution memory allowance only; model/prompt/dtype/generation unchanged."""
    def __init__(self,contract,acceptance=None):
        super().__init__(contract);self.execution_evidence={};self.early_acceptance=None
        if acceptance is not None:
            proof=read(acceptance)
            require(proof['contract_id']==canonical(contract) and proof['mode']=='early' and proof['batch_size']==16,
                    'Early-stop proof has a different contract or batch')
            require(proof['prefix_source_sha256']==sha(ROOT/'src/glare_prosody_prefix_reject.py') and
                    sha(proof['report'])==proof['report_sha256'], 'Early-stop acceptance source/report changed')
            report=read(proof['report'])
            candidate=next(r for r in report['results'] if r['mode']=='early' and r['batch']==16)
            require(report['stage']=='complete' and report['native_repeat_equal'] and report['samples']>=64 and
                    candidate['preserves_admission_and_unstopped_outputs'] and not candidate['error'],
                    'Early-stop paired verification did not pass')
            self.early_acceptance=proof
            self.execution_evidence=dict(mode='irreversible_prefix_rejection',acceptance=str(acceptance),
                acceptance_sha256=sha(acceptance),prefix_source_sha256=proof['prefix_source_sha256'])

    def load(self):
        fresh=self.analyzer is None
        super().load()
        if fresh:
            import torch
            free,total=torch.cuda.mem_get_info()
            cap=min(32*1024**3,torch.cuda.memory_reserved()+free-16*1024**3)
            require(cap>=24*1024**3,'Insufficient headroom to preserve16GiB for other GPU jobs')
            torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
            self.memory_allowance_gib=cap/1024**3
            if self.early_acceptance:
                from glare_prosody_prefix_reject import install
                self._restore_early_prefix=install(self)


def tracks_for(plan):
    with connect(plan) as db:
        for row in db.execute('SELECT data FROM tracks ORDER BY track_id'):
            track=json.loads(row['data'])
            indices=[r['chunk_index'] for r in db.execute('SELECT chunk_index FROM chunks WHERE track_id=? ORDER BY chunk_index',(track['track_id'],))]
            yield track,indices


def prepare_source(runtime,cache,track,indices,retry_execution=False):
    tick=time.monotonic();needed=[];reused=Counter()
    for i in indices:
        if retry_execution:cache.retry_execution(track,i)
        old=cache.load(track,i)
        if old is None:needed.append(i)
        else:reused[old['analysis']['status']]+=1
    result=dict(track=track,indices=needed,reused=dict(reused),attempts=[],wave=None,error=None)
    if needed:
        for attempt in range(3):
            try:
                # Bound decoded16k float32 output before whole-track official decoding.
                import soundfile as sf
                info=sf.info(track['audio']['path'])
                require(info.frames/info.samplerate*16000*4 <= 512*1024**2,
                        'One source exceeds512MiB waveform bound; explicit rescheduling required')
                result['wave']=runtime.waveform(track);break
            except IntegrityError:raise
            except (OSError,ValueError,RuntimeError) as exc:
                result['attempts'].append(dict(attempt=attempt,error=repr(exc),time=time.time()))
        if result['wave'] is None:result['error']='Cannot decode source after3 attempts'
    result['prepare_s']=time.monotonic()-tick
    return result


def run_engine(plan,p,runtime,cache,limit=None,loaders=2,batch_size=16,queue_size=64,retry_execution=False,write_queue=0):
    """CPU producer owns no GPU; sole consumer writes audited terminal receipts."""
    state=dict(stage='running',pid=os.getpid(),started=time.time(),new=0,reused=0,
        total_chunks=p['chunks'],total_tracks=p['tracks'],counts={},new_counts={},
        source_prepare_s=0.,queue_wait_s=0.,inference_s=0.,commit_s=0.,
        native_batches=0,inference_chunks=0,batch_sizes={},sources_prepared=0,
        cpu_loaders=loaders,batch_size=batch_size,queue_bound_chunks=queue_size,
        execution_code_sha256=sha(__file__),contract_id=cache.contract_id,
        execution_overlay=getattr(runtime,'execution_evidence',{}),scheduled_new=0,write_mode='async' if write_queue else 'sync')
    from glare_async_writes import BoundedWriter
    writer=BoundedWriter(write_queue) if write_queue else None
    writes=[]
    stopping=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stopping.set())
    signal.signal(signal.SIGINT,lambda *_:stopping.set())
    channel=queue.Queue(queue_size);counts=Counter();new_counts=Counter();sizes=Counter();last=[0.]
    effective_batch=[batch_size]
    def should_stop():return stopping.is_set() or (plan/'STOP').exists()
    def put(message):
        while not should_stop():
            try:channel.put(message,timeout=.5);return True
            except queue.Full:pass
        return False
    def producer():
        prepared=ThreadPoolExecutor(max_workers=loaders);futures=[];sources=None
        try:
            sources=iter(tracks_for(plan));finished=False
            while not should_stop():
                while len(futures)<loaders and not finished:
                    try:track,indices=next(sources)
                    except StopIteration:finished=True;break
                    futures.append(prepared.submit(prepare_source,runtime,cache,track,indices,retry_execution))
                if not futures:break
                result=futures.pop(0).result();track=result['track'];wave=result['wave']
                if not put(('source',result['reused'],result['prepare_s'])):return
                for index in result['indices']:
                    if should_stop():return
                    if result['error'] is not None:
                        message=('error',track,index,'input_error_exhausted',result['error'],result['attempts'])
                    elif (index+1)*CHUNK_SAMPLES>len(wave):
                        message=('error',track,index,'uncovered_source_tail',f'Source samples={len(wave)}',[])
                    else:
                        # Copy just38400 samples; queue never retains its parent full waveform.
                        message=('chunk',track,index,wave[index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES].copy())
                    if not put(message):return
                del wave,result
            put(('end',))
        except BaseException as exc:
            put(('fatal',exc,traceback.format_exc()))
        finally:
            if sources is not None:sources.close()
            for future in futures:future.cancel()
            prepared.shutdown(wait=True,cancel_futures=True)
    def pulse(force=False):
        now=time.time()
        if not force and now-last[0]<3:return
        state.update(counts=dict(counts),new_counts=dict(new_counts),batch_sizes=dict(sizes),heartbeat=now,
                     elapsed_s=now-state['started'],resolved_chunks=state['new']+state['reused'])
        state['new_chunks_per_s']=state['new']/max(.001,state['elapsed_s'])
        state['steady_inference_chunks_per_s']=state['inference_chunks']/max(.001,state['inference_s'])
        state['mean_batch_size']=state['inference_chunks']/max(1,state['native_batches'])
        if writer:state['async_writes']=writer.metrics()
        remaining=max(0,p['chunks']-state['resolved_chunks'])
        state['eta_seconds']=remaining/state['new_chunks_per_s'] if state['new']>=64 and state['new_chunks_per_s'] else None
        save(plan/'status.json',state);last[0]=now
    def mark(record):
        status=record['analysis']['status'];counts[status]+=1;new_counts[status]+=1;state['new']+=1
        if record.get('generation',{}).get('early_reject'):
            state['early_rejected_chunks']=state.get('early_rejected_chunks',0)+1
    def collect_writes(wait=False):
        while writes and (wait or writes[0].done()):mark(writes.pop(0).result())
    def commit(fn,*args):
        state['scheduled_new']+=1
        if writer:writes.append(writer.submit(fn,*args))
        else:mark(fn(*args))
    def infer(items,attempts=()):
        waves=[item[2] for item in items];runtime.load()
        state['memory_allowance_gib']=getattr(runtime,'memory_allowance_gib',None)
        history=list(attempts)
        for attempt in range(3):
            oom=False
            tick=time.monotonic()
            try:output=runtime.infer(waves)
            except IntegrityError:raise
            except Exception as exc:
                import torch
                if not isinstance(exc,(torch.cuda.OutOfMemoryError,OSError,TimeoutError)):raise
                history.append(dict(attempt=attempt,error=repr(exc),time=time.time()))
                oom=isinstance(exc,torch.cuda.OutOfMemoryError)
            else:
                state['inference_s']+=time.monotonic()-tick
                state['native_batches']+=1;state['inference_chunks']+=len(items);sizes[len(items)]+=1
                require(len(output)==len(items),'Packed inference result cardinality mismatch')
                tick=time.monotonic();members=[cache.chunk_id(track,index) for track,index,_ in items]
                for (track,index,wave),(raw,generation,native) in zip(items,output):
                    generation=dict(generation,execution_code_sha256=state['execution_code_sha256'],batch_members=members,
                                    execution_overlay=getattr(runtime,'execution_evidence',{}))
                    if generation.get('early_reject'):
                        from glare_prosody_prefix_reject import impossible_prefix
                        require(impossible_prefix(raw) is not None and not audit_raw(raw,generation['truncated'])['accepted'],
                                'Early stop cannot produce an admissible target or an unproven rejection')
                    commit(cache.commit,track,index,raw,generation,native,waveform_sha(wave),history)
                collect_writes()
                state['commit_s']+=time.monotonic()-tick;return
            # Outside except: the original traceback must release failed CUDA tensors
            # before subdivision; empty_cache cannot free tensors retained by it.
            gc.collect();torch.cuda.empty_cache()
            if oom and len(items)>1:
                state['oom_subdivisions']=state.get('oom_subdivisions',0)+1
                effective_batch[0]=min(effective_batch[0],max(1,len(items)//2))
                state['effective_batch_size']=effective_batch[0]
                mid=len(items)//2;infer(items[:mid],history);infer(items[mid:],history);return
            if oom and attempt==2:
                raise RuntimeError('Single-chunk OOM persists; abort resource failure instead of excluding the whole corpus')
        for track,index,_ in items:
            commit(cache.error,track,index,'inference_error_exhausted','Temporary execution failure after3 attempts',history)
    worker=threading.Thread(target=producer,name='source-prefetch',daemon=True)
    pending=[];pulse(True);worker.start()
    try:
        done=False
        while not done and not should_stop():
            collect_writes()
            if limit and state['scheduled_new']>=limit:break
            tick=time.monotonic()
            try:item=channel.get(timeout=1.)
            except queue.Empty:pulse();continue
            state['queue_wait_s']+=time.monotonic()-tick
            kind=item[0]
            if kind=='source':
                counts.update(item[1]);state['reused']+=sum(item[1].values())
                state['source_prepare_s']+=item[2];state['sources_prepared']+=1
            elif kind=='error':
                _,track,index,status,error,attempts=item
                if limit and state['scheduled_new']+len(pending)>=limit:
                    if pending:infer(pending);pending=[]
                    break
                commit(cache.error,track,index,status,error,attempts)
            elif kind=='fatal':raise RuntimeError('Source producer failed:\n'+item[2]) from item[1]
            elif kind=='end':done=True
            elif kind=='chunk':pending.append(item[1:])
            else:raise IntegrityError('Unknown queue message')
            target=effective_batch[0] if not limit else min(effective_batch[0],max(1,limit-state['scheduled_new']))
            if len(pending)>=target or (done and pending):
                infer(pending);pending=[];pulse()
        if writer:state['stage']='draining_writes';pulse(True)
        collect_writes(True)
        if should_stop():
            state['stage']='stopped'
        elif limit and state['new']>=limit:
            state['stage']='bounded_run_complete'
        else:
            require(state['new']+state['reused']==p['chunks'],'Full run ended with uncovered unrecorded dependencies')
            state['stage']='complete'
    except BaseException as exc:
        state.update(stage='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        stopping.set();worker.join(timeout=60)
        try:
            collect_writes(True)
        except BaseException as exc:
            state.update(stage='failed',error=repr(exc));raise
        finally:
            try:
                if writer:writer.close()
            except BaseException as exc:
                state.update(stage='failed',error=repr(exc));raise
            finally:
                state.update(finished=time.time(),producer_drained=not worker.is_alive());pulse(True)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=BASE/'full')
    parser.add_argument('--limit',type=int);parser.add_argument('--loaders',type=int,default=2,choices=[1,2,3,4])
    parser.add_argument('--batch-size',type=int,default=16,choices=range(1,17));parser.add_argument('--queue-size',type=int,default=64)
    parser.add_argument('--retry-execution',action='store_true')
    parser.add_argument('--early-reject-acceptance',type=Path)
    parser.add_argument('--official-resources',action='store_true')
    parser.add_argument('--write-queue',type=int,default=0,help='0 synchronous; bounded CPU commit operations otherwise')
    a=parser.parse_args();a.plan=a.plan.resolve()
    require(a.limit is None or a.limit>0,'Invalid limit');require(a.batch_size<=a.queue_size<=128,'Invalid queue bound')
    require(0<=a.write_queue<=256,'Invalid write queue bound')
    require(a.early_reject_acceptance is None or a.batch_size==16,'Early rejection validated only at batch16')
    require(not (a.official_resources and a.early_reject_acceptance),'Official complete generation cannot use early stop')
    p=protocol(a.plan);cache=ChunkCache(p['cache_root'],p['extractor'])
    with ExitStack() as guards:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock',cache.root/'mutation.lock'):
            guards.enter_context(lock(path,False))
        if a.official_resources:
            from glare_official_capacity_runtime import CapacityOfficialRuntime
            runtime=CapacityOfficialRuntime(p['extractor'])
        else:runtime=PackedRuntime(p['extractor'],a.early_reject_acceptance)
        state=run_engine(a.plan,p,runtime,cache,a.limit,a.loaders,a.batch_size,a.queue_size,a.retry_execution,a.write_queue)
        print(json.dumps(state),flush=True)


if __name__=='__main__':main()
