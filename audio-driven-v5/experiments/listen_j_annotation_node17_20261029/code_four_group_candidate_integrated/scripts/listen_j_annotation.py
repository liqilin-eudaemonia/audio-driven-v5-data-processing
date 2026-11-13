#!/usr/bin/env python3
"""J result ingestion, bounded frame priming and persistent GPU annotation."""
import argparse
from collections import deque
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,acquire,locked,committed_job,JobQueue,result_path,digest


def ingest(args):
    work=args.work;owner=acquire(work/'ingest.lock',False)
    if owner is None:return
    manifest=read(args.upstream/'manifest.json')
    imported=set();imported_order=[]
    journal=work/'imported.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            pair=json.loads(line)['pair']
            if pair not in imported:imported.add(pair);imported_order.append(pair)
    stats=read(work/'ingest_status.json') if (work/'ingest_status.json').exists() else {'sources':0,'ready_jobs':0,'ready_clips':0,'excluded_clips':0}
    # The append journal is durable per source; the summary is only periodic.
    for pair in imported_order[stats['sources']:]:
        job=read(work/'jobs'/f'{pair}.json');stats['sources']+=1
        stats['ready_jobs']+=bool(job['rows']);stats['ready_clips']+=len(job['rows']);stats['excluded_clips']+=len(job['excluded'])
    waiting=[item for item in manifest['items'] if item['pair'] not in imported]
    while waiting:
        pending=[]
        for item in waiting:
            pair=item['pair'];receipt=args.upstream/'results'/pair/'complete.json'
            if not receipt.exists():pending.append(item);continue
            try:
                job=committed_job(args.upstream,item,manifest['contract'])
                write(work/'jobs'/f'{pair}.json',job)
                if job['rows']:
                    with (work/'ready.jsonl').open('a') as f:
                        f.write(json.dumps({'job_id':pair,'count':len(job['rows'])})+'\n');f.flush()
                    stats['ready_jobs']+=1;stats['ready_clips']+=len(job['rows'])
                stats['sources']+=1;stats['excluded_clips']+=len(job['excluded'])
                with journal.open('a') as f:f.write(json.dumps({'pair':pair,'hash':job['source_result_sha256']})+'\n')
                imported.add(pair)
                (work/'ingest_errors'/f'{pair}.json').unlink(missing_ok=True)
            except Exception as e:
                write(work/'ingest_errors'/f'{pair}.json',{'error':repr(e),'at':time.time()});pending.append(item)
            if stats['sources']%50==0:
                stats.update(at=time.time(),remaining_sources=len(manifest['items'])-len(imported));write(work/'ingest_status.json',stats)
        stats.update(at=time.time(),remaining_sources=len(pending));write(work/'ingest_status.json',stats)
        upstream=read(args.upstream/'status.json')
        if upstream['state'].startswith('completed'):
            failed=read(args.upstream/'queue.json').get('failed',{})
            pending=[x for x in pending if x['pair'] not in failed]
            stats['upstream_failed_sources']=list(failed)
        stats['remaining_sources']=len(pending)
        waiting=pending
        if waiting:time.sleep(60)
    write(work/'ingest_complete.json',dict(stats,at=time.time(),total_sources=len(manifest['items'])));owner.close()


def prime(args):
    owner=acquire(args.work/'prime.lock',False)
    if owner is None:return
    from listen_benchmark.visual_native8 import prepare_media
    from listen_benchmark.visual_media_cache import cached_media
    from listen_visual_native8_nod import decode_slot
    cfg=read(args.work/'config.json');seen=set();completed=[]
    started=time.time()
    def run(row):
        def decode():
            tick=time.monotonic()
            with decode_slot(args.work/'resources',cfg['decode_slots']):
                acquired=time.monotonic();media=prepare_media(row['source'])
            return media,{'decode_s':time.monotonic()-acquired,'decode_lock_wait_s':acquired-tick}
        try:
            _,metrics=cached_media(row['source'],cfg['warm_cache'],decode)
            return {'sample_id':row['sample_id'],'status':'ok',**metrics}
        except Exception as e:return {'sample_id':row['sample_id'],'status':'error','error':repr(e)}
    while len(seen)<cfg['warm_samples']:
        journal=args.work/'ready.jsonl'
        if not journal.exists():time.sleep(5);continue
        rows=[]
        for line in journal.read_text().splitlines():
            job=read(args.work/'jobs'/f"{json.loads(line)['job_id']}.json")
            for row in job['rows']:
                if row['sample_id'] not in seen:
                    seen.add(row['sample_id'])
                    if not result_path(args.work,'vlm',row['sample_id']).exists():rows.append(row)
                    if len(seen)>=cfg['warm_samples']:break
            if len(seen)>=cfg['warm_samples']:break
        if rows:
            with ThreadPoolExecutor(max_workers=cfg['warm_workers']) as pool:
                for result in pool.map(run,rows):
                    completed.append(result)
                    write(args.work/'prime_status.json',{'completed':len(completed),'target':cfg['warm_samples'],
                        'at':time.time(),'last':result,'elapsed_s':time.time()-started})
        if (args.work/'ingest_complete.json').exists():break
        if len(seen)<cfg['warm_samples']:time.sleep(5)
    write(args.work/'prime_complete.json',{'completed':len(completed),'results':completed,'wall_s':time.time()-started});owner.close()


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from listen_benchmark.visual_native8 import prepare_media,prepare_input,MM_SIZE
    from listen_benchmark.shared_video_inputs import prepare_shared_inputs
    from listen_benchmark.engine_input_renderer import EngineInputRenderer,estimate_render_frames
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.full_raw import views
    from listen_benchmark.visual_media_cache import cached_media,cache_file
    from listen_benchmark.visual_batch import annotate_group
    from listen_benchmark.visual_schedule import prefetched
    from listen_benchmark.simple_timeline import response_schema
    from listen_visual_native8_nod import decode_slot
    torch.set_num_threads(2)
    work=args.work;name=f'gpu-{args.gpu}';owner=acquire(work/'workers'/f'{name}.lock',False)
    if owner is None:return
    execution=read(work/'selected.json');config=read(ROOT/'configs/listen_visual_native8_nod_v1.json')
    cfg=read(work/'config.json');queue=JobQueue(work,'vlm');stop=threading.Event()
    from listen_benchmark.execution_profile import execution_for_gpu,engine_graph_kwargs
    execution=execution_for_gpu(execution,cfg,args.gpu)
    from listen_benchmark.j_annotation import ensure_decode_fd_capacity
    fd_limits=ensure_decode_fd_capacity(execution['decode_workers'],execution['decode_slots'])
    render_workers=execution.get('renderer_workers',0)
    if render_workers not in (0,2,4):raise ValueError('Renderer trial supports disabled, two or four workers')
    rolling_enabled=cfg.get('rolling_inference',False) and (
        'rolling_canary_gpus' not in cfg or args.gpu in cfg['rolling_canary_gpus'])
    annotation_groups=execution.get('rolling_annotation_groups',2 if rolling_enabled else 1)
    execution=dict(execution,fast_frame_crop=cfg.get('visual_fast_frame_crop',False),
                   shared_frame_arrays=cfg.get('shared_frame_arrays',False),
                   engine_metrics_enabled=True,renderer_workers=render_workers,
                   mm_processor_cache_gb=0 if render_workers else 4,prefix_caching=True,
                   bounded_source_groups=True,fd_limits_at_start=fd_limits,
                   rolling_inference=rolling_enabled,rolling_annotation_groups=annotation_groups,
                   worker_id=name,worker_pid=os.getpid())
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    state={'pid':os.getpid(),'gpu':args.gpu,'started_at':time.time(),'stage':'model_loading',
           'completed_clips':0,'failed_clips':0,'completed_jobs':0,'execution':execution,
           'revision':12 if annotation_groups==4 else (11 if execution.get('decode_backend')=='process' else (10 if render_workers==4 else (9 if execution.get('decode_graph_capture_sizes') else (8 if rolling_enabled else 7))))}
    def pulse():
        while not stop.wait(10):write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
    threading.Thread(target=pulse,daemon=True).start()
    write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
    processor=AutoProcessor.from_pretrained(config['model'],trust_remote_code=True,local_files_only=True)
    llm=LLM(model=config['model'],dtype='bfloat16',gpu_memory_utilization=execution['memory'],
        max_model_len=config['max_model_len'],max_num_seqs=execution['batch_size'],
        max_num_batched_tokens=execution['tokens'],enable_chunked_prefill=True,enable_prefix_caching=True,
        limit_mm_per_prompt={'image':0,'video':1},skip_mm_profiling=True,
        mm_processor_kwargs={'size':MM_SIZE},seed=config['seed'],
        **engine_graph_kwargs(execution),
        disable_log_stats=False,
        **({'renderer_num_workers':render_workers,'mm_processor_cache_gb':0} if render_workers else {}))
    engine_renderer=EngineInputRenderer(llm.renderer) if render_workers else None
    from listen_benchmark.process_decode import ProcessMediaDecoder
    process_decoder=(ProcessMediaDecoder(cfg['warm_cache'],work/'resources',execution['fast_frame_crop'])
                     if execution.get('decode_backend')=='process' else None)
    def make_params(duration):
        return SamplingParams(temperature=config['temperature'],max_tokens=config['max_tokens'],
            structured_outputs=StructuredOutputsParams(json=response_schema(duration)))
    from listen_benchmark.j_annotation import bounded_claim_groups
    def decode(row,slots):
        def raw_decode():
            tick=time.monotonic()
            with decode_slot(work/'resources',slots):
                acquired=time.monotonic();media=prepare_media(row['source'],
                    frame_crop=crop_original_frame if execution['fast_frame_crop'] else None)
            return media,{'decode_s':time.monotonic()-acquired,'decode_lock_wait_s':acquired-tick}
        try:
            # Bounded streaming cache: materialize only this group's frames;
            # consumed cache files are removed after durable result writes.
            media,metrics=cached_media(row['source'],cfg['warm_cache'],raw_decode)
            return row,media,metrics
        except Exception as e:return row,None,{'error':repr(e)}
    @contextmanager
    def decoder_batch(rows,workers,slots):
        if process_decoder is not None:
            yield process_decoder.map(rows,workers,slots)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                yield pool.map(lambda row:decode(row,slots),rows)
    def load_group(group):
        if not group['rows']:return group,[]
        state.update(preparing_jobs=[j for j,_,_ in group['claims']],
                     preparing_clips=len(group['rows']))
        tuning=read(work/'resource_tuning.json') if (work/'resource_tuning.json').exists() else execution
        workers=tuning['decode_workers'];slots=tuning['decode_slots']
        state.update(decode_workers_active=workers,decode_slots_active=slots)
        precompile=tuning.get('precompile_inputs',False)
        render_frame_estimate=estimate_render_frames(group['rows']) if engine_renderer is not None else 0
        render_frame_limit=12288
        if engine_renderer is not None and render_frame_estimate<=render_frame_limit:
            if not precompile or not execution['shared_frame_arrays']:
                raise ValueError('Engine renderer requires shared precompiled inputs')
            # Decode jobs run ahead while completed clips enter the official
            # async renderer. At most four clip render futures are pending;
            # the whole current/next group remains bounded by the existing queue.
            started=time.monotonic();loaded=[];pending=deque();array_prepare_s=0.
            def finish_one():
                item,metrics=pending.popleft()
                metrics['engine_render_s']=engine_renderer.attach(item)
            with decoder_batch(group['rows'],workers,slots) as decoded:
                for row,media,metrics in decoded:
                    loaded.append((row,media,metrics))
                    if media is None or stop.is_set():continue
                    tick=time.monotonic();media['_prepared_inputs']=prepare_shared_inputs(processor,media)
                    metrics['background_prepare_s']=time.monotonic()-tick
                    metrics['shared_frame_arrays']=True
                    array_prepare_s+=metrics['background_prepare_s']
                    pending.append((engine_renderer.submit(media['_prepared_inputs']),metrics))
                    if len(pending)>=4:finish_one()
            while pending:finish_one()
            group['prefetch_metrics']={'total_prepare_wall_s':time.monotonic()-started,
                'array_prepare_cpu_wall_s':array_prepare_s,'precompiled_inputs':True,
                'shared_frame_arrays':True,'renderer_workers':render_workers,'max_pending_render_clips':4,
                'decode_and_renderer_overlap':True,'render_frame_estimate':render_frame_estimate,
                'render_frame_limit':render_frame_limit}
            state.update(precompiled_inputs_active=True,engine_prerender_active=True)
            return group,loaded
        tick=time.monotonic()
        with decoder_batch(group['rows'],workers,slots) as decoded:
            loaded=list(decoded)
        decode_wall=time.monotonic()-tick;tick=time.monotonic()
        if precompile:
            # CPU-only input assembly overlaps the preceding group's GPU work.
            # Use the exact same builder, intervals, arrays and chat template.
            for row,media,metrics in loaded:
                if media is None:continue
                started=time.monotonic()
                if execution['shared_frame_arrays']:
                    media['_prepared_inputs']=prepare_shared_inputs(processor,media)
                else:
                    media['_prepared_inputs']={}
                    for view,start,end in views(media[8]['duration_s']):
                        media['_prepared_inputs'][view]={'interval':(start,end),
                            'input':prepare_input(processor,media,start,end,view)}
                metrics['background_prepare_s']=time.monotonic()-started
                metrics['shared_frame_arrays']=execution['shared_frame_arrays']
        group['prefetch_metrics']={'decode_wall_s':decode_wall,'input_prepare_wall_s':time.monotonic()-tick,
            'precompiled_inputs':precompile,'shared_frame_arrays':precompile and execution['shared_frame_arrays'],
            'renderer_workers':0,'render_frame_estimate':render_frame_estimate,
            'renderer_skip_reason':'group_frame_memory_limit' if engine_renderer is not None else 'disabled'}
        state.update(precompiled_inputs_active=precompile,engine_prerender_active=False)
        return group,loaded
    config_hash=digest((ROOT/'configs/listen_visual_native8_nod_v1.json').read_bytes())
    def run_rolling():
        from listen_benchmark.rolling_inference import VllmStepDriver,RollingBatchEngine,rolling_groups
        engine=RollingBatchEngine(VllmStepDriver(llm),execution['batch_size'],execution['frames'])
        def load(group):
            if not group['claims']:time.sleep(2)
            return load_group(group)
        def annotate(group,loaded,proxy):
            valid=[(row,media) for row,media,_ in loaded if media is not None]
            errors={row['sample_id']:metrics['error'] for row,media,metrics in loaded if media is None}
            results=annotate_group(valid,processor,proxy,make_params,work/'vlm',
                execution['batch_size'],execution['frames'],write,config_hash)
            return results,errors
        def status(active):
            state.update(stage='annotation' if active else 'preparing_next_group',
                active_jobs=list(dict.fromkeys(j for group,_ in active for j,_,_ in group['claims'])),
                active_clips=sum(len(loaded) for _,loaded in active),
                rolling_pending_requests=len(engine.pending),rolling_pending_frames=engine.frames,
                rolling_peak_requests=engine.peak_requests,rolling_peak_frames=engine.peak_frames,
                rolling_refills_while_busy=engine.refills_while_busy)
        with closing(bounded_claim_groups(queue,name,work,execution['clip_batch_size'],stop)) as groups_iter, \
             closing(rolling_groups(groups_iter,load,annotate,engine,stop,status,
                                    max_active_groups=annotation_groups,progress=state.update)) as ready:
            for group,loaded,(results,errors) in ready:
                # Commit in source-group order. A later chunk may finish GPU
                # work first, but cannot prematurely finalize its source.
                for sid,error in errors.items():
                    state['failed_clips']+=1
                    write(work/'input_errors'/f'{sid}.json',{'error':error,'at':time.time()})
                by_id={row['sample_id']:row for row,_,_ in loaded}
                by_metrics={row['sample_id']:metrics for row,_,metrics in loaded}
                for result in results:
                    sid=result['sample_id']
                    result.update(source=by_id[sid]['source'],execution=execution,
                        completed_at=time.time(),training_admitted=False,
                        input_performance=by_metrics[sid],prefetch_group_performance=group['prefetch_metrics'])
                    write(result_path(work,'vlm',sid),result)
                    (work/'vlm/partial'/f'{sid}.json').unlink(missing_ok=True)
                    cache_file(by_id[sid]['source'],cfg['warm_cache']).unlink(missing_ok=True)
                    state['completed_clips']+=1
                for job_id,handle,job in group['finalize_claims']:
                    missing=[r['sample_id'] for r in job['rows'] if not result_path(work,'vlm',r['sample_id']).exists()]
                    if missing:queue.finish(job_id,error={'missing':missing,'errors':errors})
                    else:
                        queue.finish(job_id,receipt={'job_id':job_id,'source_result_sha256':job['source_result_sha256'],
                            'clips':len(job['rows']),'completed_at':time.time()});state['completed_jobs']+=1
                    handle.close()
                state.update(last_progress_at=time.time())
                write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
                del loaded,results
    state['stage']='preparing_first_group'
    try:
        if rolling_enabled:
            run_rolling()
            return
        with closing(bounded_claim_groups(queue,name,work,execution['clip_batch_size'],stop)) as groups_iter, \
             closing(prefetched(groups_iter,load_group,1)) as ready:
            for _,(group,loaded) in ready:
                if stop.is_set():break
                claims=group['claims'];errors={}
                try:
                    if not claims:
                        state['stage']='waiting_for_data';time.sleep(2);continue
                    state.update(stage='annotation',active_jobs=[j for j,_,_ in claims],active_clips=len(loaded))
                    valid=[]
                    for row,media,metrics in loaded:
                        if media is None:
                            errors[row['sample_id']]=metrics['error'];state['failed_clips']+=1
                            write(work/'input_errors'/f"{row['sample_id']}.json",dict(metrics,at=time.time()))
                        else:valid.append((row,media))
                    for offset in range(0,len(valid),execution['clip_batch_size']):
                        subset=valid[offset:offset+execution['clip_batch_size']]
                        results=annotate_group(subset,processor,llm,make_params,work/'vlm',
                            execution['batch_size'],execution['frames'],write,config_hash)
                        by_id={r['sample_id']:r for r,_ in subset}
                        by_metrics={row['sample_id']:metrics for row,_,metrics in loaded}
                        for result in results:
                            sid=result['sample_id'];result.update(source=by_id[sid]['source'],execution=execution,
                                completed_at=time.time(),training_admitted=False,
                                input_performance=by_metrics[sid],prefetch_group_performance=group['prefetch_metrics'])
                            write(result_path(work,'vlm',sid),result)
                            (work/'vlm/partial'/f'{sid}.json').unlink(missing_ok=True)
                            cache_file(by_id[sid]['source'],cfg['warm_cache']).unlink(missing_ok=True)
                            state['completed_clips']+=1
                    for job_id,handle,job in group['finalize_claims']:
                        missing=[r['sample_id'] for r in job['rows'] if not result_path(work,'vlm',r['sample_id']).exists()]
                        if missing:queue.finish(job_id,error={'missing':missing,'errors':errors})
                        else:
                            queue.finish(job_id,receipt={'job_id':job_id,'source_result_sha256':job['source_result_sha256'],
                                'clips':len(job['rows']),'completed_at':time.time()});state['completed_jobs']+=1
                    state.update(stage='preparing_next_group',last_progress_at=time.time())
                    write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
                except BaseException as e:
                    for job_id,_,_ in claims:queue.finish(job_id,error=repr(e))
                    raise
                finally:
                    for _,handle,_ in group['finalize_claims']:handle.close()
                    del loaded
    except BaseException as e:
        state.update(stage='error',error=repr(e));raise
    finally:
        stop.set()
        if process_decoder is not None:process_decoder.close()
        if engine_renderer is not None:engine_renderer.close()
        write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()));owner.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('ingest','prime','worker'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
    args.work=args.work.resolve();args.upstream=args.upstream.resolve()
    globals()[args.command](args)


if __name__=='__main__':main()
