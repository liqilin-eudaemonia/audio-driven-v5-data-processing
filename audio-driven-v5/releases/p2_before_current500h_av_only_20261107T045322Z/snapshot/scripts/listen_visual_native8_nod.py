#!/usr/bin/env python3
"""Confirmed visual annotation: native video8 + independently sourced nod candidates.

`vlm` starts GPU inference explicitly. `export` only reads existing model results.
CPU nod extraction continues to use run_glare_nodding_pilot.py; see the handoff.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from contextlib import contextmanager, closing, ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
CONFIG=ROOT/'configs/listen_visual_native8_nod_v1.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def selected(manifest):
    data=json.loads(manifest.read_text())
    rows=data if isinstance(data,list) else [{'sample_id':sid} for sid in data['ids']]
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(Path(s).name!=s for s in ids):raise ValueError('Invalid or duplicate sample IDs')
    return rows

@contextmanager
def lock(out,name):
    path=out/'locks'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

@contextmanager
def decode_slot(out, slots):
    """Host-wide bounded decoding; benchmark directories share these lock files."""
    with ExitStack() as stack:
        handles=[]
        for i in range(slots):
            path=out/'locks'/('decode-'+socket.gethostname()+(f'-{i}' if slots>1 else ''))
            path.parent.mkdir(parents=True,exist_ok=True)
            handles.append(stack.enter_context(path.open('a')))
        while True:
            for handle in handles:
                try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                try:yield
                finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
                return
            time.sleep(.05)

def export(args):
    from listen_benchmark.simple_timeline import validate_timeline
    from listen_benchmark.visual_native8 import observation_prompt
    cfg=json.loads(CONFIG.read_text());rows=selected(args.manifest);out=[]
    for row in rows:
        sid=row['sample_id'];vp=args.vlm_results/f'{sid}.json';np=args.nod_results/f'{sid}.json'
        raw=json.loads(vp.read_text());nod=json.loads(np.read_text())
        assert raw['sample_id']==sid and nod['id']==sid and nod['status']=='ok'
        duration=raw['duration_s'];assert abs(duration-nod['duration'])<1e-6
        records=[r for r in raw['records'] if r['arm']=='video8']
        assert records and len({r['view'] for r in records})==len(records)
        for r in records:
            assert r['prompt']==observation_prompt(r['end']-r['start']), 'Unexpected prompt; refuse silently mixing versions'
            assert r['native']['do_resize'] is False and r['native']['do_sample_frames'] is False
            assert r['media']['video']['requested_fps']==8 and r['native']['audio_input'] is False
        full=next(r for r in records if r['view']=='full')
        assert full['status']=='valid' and full['finish_reason']=='stop'
        validate_timeline(full['parsed'],duration)
        assert all(e['modality']=='visual' for e in full['parsed']['events'])
        # Pose metadata was validated against the cache crop during nod extraction.
        media=json.loads((Path(nod['video']).parent/'media.json').read_text())
        vm=full['media']['video']
        assert media['sample_id']==sid and media['video_transform']==vm['video_transform']
        assert abs(media['source_start_s']-vm['source_time_origin_s'])<=1/48000
        assert nod['fps']==cfg['nodding']['input_fps']
        # Reject accidental use of a different recall profile.
        nod_protocol=json.loads((args.nod_results.parent/'protocol.json').read_text())
        expected={k:v for k,v in cfg['nodding'].items() if k not in ('input_fps','profile')}
        assert nod_protocol['profiles']['recall']==expected
        events=[]
        for i,event in enumerate(nod['profiles']['recall']['events']):
            assert 0<=event['start']<event['end']<=duration
            events.append(dict(event,event_id=f'nod-{i+1:04d}',type='nodding',modality='visual',
                               description='Head nodding',description_source='detector_label_not_vlm_quote',
                               source='GLARE_nodding_v2_recall',status='candidate'))
        result={'schema':'listen-visual-native8-nod-v1','sample_id':sid,'duration_s':duration,
                'config_sha256':sha(CONFIG),'sources':{'vlm_file':str(vp.resolve()),'vlm_sha256':sha(vp),
                    'nod_file':str(np.resolve()),'nod_sha256':sha(np),'nod_protocol_sha256':sha(args.nod_results.parent/'protocol.json')},
                'vlm_observations':copy.deepcopy(records),'vlm_full_events':copy.deepcopy(full['parsed']['events']),
                'nod_candidates':events,'disabled_detector_types':list(cfg['supplement_disabled']),
                'status':'candidate_bundle_not_final_training_labels','training_admitted':False}
        save(args.output/'results'/f'{sid}.json',result);out.append(result)
    with (args.output/'bundles.jsonl').open('w') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'samples':len(out),'vlm_full_events':sum(len(r['vlm_full_events']) for r in out),
            'nod_candidates':sum(len(r['nod_candidates']) for r in out),'automatic_headshake_supplements':0,
            'invalid_windows_preserved':sum(x['status']!='valid' for r in out for x in r['vlm_observations']),
            'config_sha256':sha(CONFIG),'manifest_sha256':sha(args.manifest),'production_modified':False}
    save(args.output/'summary.json',report);print(json.dumps(report))

def annotate_sample(args, processor, llm, row, whole, make_params):
    """Persist complete batches; only publish a result after all views exist."""
    from listen_benchmark.visual_native8 import prepare_input
    from listen_benchmark.full_raw import views
    from listen_benchmark.simple_timeline import validate_timeline
    from listen_benchmark.visual_schedule import window_batches
    sid=row['sample_id'];duration=whole[8]['duration_s']
    partial=args.output/'partial'/f'{sid}.json'
    checkpoint=json.loads(partial.read_text()) if partial.exists() else {'sample_id':sid,'records':[]}
    if checkpoint['sample_id']!=sid:raise ValueError('Checkpoint sample mismatch')
    records=checkpoint['records'];expected=views(duration)
    expected_by_id={view:(start,end) for view,start,end in expected}
    completed={r['view'] for r in records}
    if len(completed)!=len(records):raise ValueError('Duplicate checkpoint views')
    for rec in records:
        if rec['view'] not in expected_by_id or (rec['start'],rec['end'])!=expected_by_id[rec['view']]:
            raise ValueError('Checkpoint window mismatch')
        if rec['arm']!='video8' or rec.get('status') not in ('valid','invalid'):
            raise ValueError('Incomplete checkpoint record')
    pending=[w for w in expected if w[0] not in completed]
    order={w[0]:i for i,w in enumerate(expected)}
    metrics={'resumed_windows':len(records),'new_windows':0,'batches':0,
             'prepare_s':0.,'generate_s':0.,'batch_sizes':[],'timing_scope':'current_attempt_only'}
    for batch in window_batches(pending,args.batch_size,args.batch_frame_budget):
        tick=time.monotonic()
        prepared=[prepare_input(processor,whole,start,end,view) for view,start,end in batch]
        params=[make_params(end-start) for _,start,end in batch]
        metrics['prepare_s']+=time.monotonic()-tick
        tick=time.monotonic()
        responses=llm.generate([item[0] for item in prepared],params,use_tqdm=False)
        elapsed=time.monotonic()-tick
        if len(responses)!=len(batch):raise RuntimeError('Incomplete inference batch; no result published')
        batch_id=f'{sid}:{batch[0][0]}'
        for (view,start,end),(_,evidence),response in zip(batch,prepared,responses):
            answer=response.outputs[0]
            rec=dict(arm='video8',view=view,start=start,end=end,raw=answer.text,finish_reason=answer.finish_reason,
                     prompt_tokens=len(response.prompt_token_ids),completion_tokens=len(answer.token_ids),
                     elapsed_s=elapsed,elapsed_scope='shared_batch_wall_time',batch_id=batch_id,
                     batch_size=len(batch),**evidence)
            try:
                rec['parsed']=json.loads(answer.text);validate_timeline(rec['parsed'],end-start)
                assert all(e['modality']=='visual' for e in rec['parsed']['events']) and answer.finish_reason=='stop'
                rec['status']='valid'
            except Exception as exc:rec.update(status='invalid',error=str(exc))
            records.append(rec)
        records.sort(key=lambda r:order[r['view']])
        metrics['generate_s']+=elapsed;metrics['new_windows']+=len(batch)
        metrics['batches']+=1;metrics['batch_sizes'].append(len(batch))
        save(partial,{'sample_id':sid,'records':records})
        del prepared,responses
    assert {r['view'] for r in records}==set(expected_by_id)
    result={'sample_id':sid,'duration_s':duration,'records':records,'config_sha256':sha(CONFIG),
            'performance':metrics}
    return result


def vlm(args):
    from listen_benchmark.visual_native8 import prepare_media,MM_SIZE
    from listen_benchmark.simple_timeline import response_schema
    from listen_benchmark.visual_schedule import assign_rows,estimated_work,prefetched
    cfg=json.loads(CONFIG.read_text());rows=selected(args.manifest)
    if not 0<=args.rank<args.shards:raise ValueError('Invalid shard rank')
    if args.batch_size<1 or args.batch_frame_budget<1 or args.max_num_batched_tokens<1:
        raise ValueError('Batch limits must be positive')
    if args.clip_batch_size<1 or args.decode_slots<1 or args.decode_workers<1:
        raise ValueError('Clip batch/decode concurrency must be positive')
    if not 0<args.memory<1:raise ValueError('memory must be between 0 and 1')
    if any('source' not in r for r in rows):raise ValueError('vlm needs full source rows, not an IDs-only manifest')
    assignments=assign_rows(rows,args.shards,args.scheduling)
    execution={'batch_size':args.batch_size,'batch_frame_budget':args.batch_frame_budget,
               'prefetch':args.prefetch,'scheduling':args.scheduling,
               'max_num_batched_tokens':args.max_num_batched_tokens,'gpu_memory_utilization':args.memory,
               'enforce_eager':True,'clip_batch_size':args.clip_batch_size,'decode_slots':args.decode_slots,
               'decode_workers':args.decode_workers,
               'prepared_media_cache':str(args.prepared_media_cache) if args.prepared_media_cache else None}
    run_cfg={'config_sha256':sha(CONFIG),'manifest_sha256':sha(args.manifest),'shards':args.shards,
             'runner_sha256':sha(Path(__file__)),'native_code_sha256':sha(ROOT/'src/listen_benchmark/visual_native8.py'),
             'schedule_code_sha256':sha(ROOT/'src/listen_benchmark/visual_schedule.py'),
             'batch_code_sha256':sha(ROOT/'src/listen_benchmark/visual_batch.py'),
             'cache_code_sha256':sha(ROOT/'src/listen_benchmark/visual_media_cache.py'),
             'execution':execution}
    with lock(args.output,'protocol.lock'):
        p=args.output/'protocol.json'
        if p.exists():assert json.loads(p.read_text())==run_cfg,'Changed run inputs/config; choose a new output'
        else:save(p,run_cfg)
    save(args.output/'schedule'/f'rank-{args.rank:03d}.json',
         {'rank':args.rank,'sample_ids':[r['sample_id'] for r in assignments[args.rank]],
          'estimated_work':sum(estimated_work(r) for r in assignments[args.rank]),'execution':execution})
    # Avoid loading a model on an empty or already completed shard.
    assigned=[r for r in assignments[args.rank] if not (args.output/'results'/f"{r['sample_id']}.json").exists()]
    if not assigned:
        print(f'rank {args.rank}: already complete',flush=True);return
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
    import torch
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    torch.set_num_threads(2)
    processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
    with lock(args.output,'model-init-'+socket.gethostname()):
        llm=LLM(model=cfg['model'],dtype='bfloat16',gpu_memory_utilization=args.memory,
                max_model_len=cfg['max_model_len'],max_num_seqs=args.batch_size,
                max_num_batched_tokens=args.max_num_batched_tokens,enable_chunked_prefill=True,
                limit_mm_per_prompt={'image':0,'video':1},skip_mm_profiling=True,
                mm_processor_kwargs={'size':MM_SIZE},enforce_eager=True,seed=cfg['seed'])
    def load(row):
        def decode():
            tick=time.monotonic()
            with decode_slot(args.output,args.decode_slots):
                acquired=time.monotonic();whole=prepare_media(row['source'])
            return whole,{'decode_s':time.monotonic()-acquired,'decode_lock_wait_s':acquired-tick}
        if args.prepared_media_cache:
            from listen_benchmark.visual_media_cache import cached_media
            return cached_media(row['source'],args.prepared_media_cache,decode)
        return decode()
    def make_params(duration):
        return SamplingParams(temperature=cfg['temperature'],max_tokens=cfg['max_tokens'],
                    structured_outputs=StructuredOutputsParams(json=response_schema(duration)))
    tick=time.monotonic()
    if args.clip_batch_size>1:
        from listen_benchmark.visual_batch import annotate_group
        groups=[assigned[i:i+args.clip_batch_size] for i in range(0,len(assigned),args.clip_batch_size)]
        def load_group(group):
            if args.decode_workers==1:return [(row,*load(row)) for row in group]
            from concurrent.futures import ThreadPoolExecutor
            # CPU-only PyAV reads; GPU inference stays in the main thread.
            with ThreadPoolExecutor(max_workers=args.decode_workers,thread_name_prefix='clip-decode') as pool:
                return list(pool.map(lambda row:(row,*load(row)),group))
        with closing(prefetched(groups,load_group,args.prefetch)) as ready:
            for _,loaded in ready:
                wait=time.monotonic()-tick
                with ExitStack() as stack:
                    for row,_,_ in sorted(loaded,key=lambda x:x[0]['sample_id']):
                        stack.enter_context(lock(args.output,'sample-'+row['sample_id']))
                    pending=[(row,whole,metrics) for row,whole,metrics in loaded
                             if not (args.output/'results'/f"{row['sample_id']}.json").exists()]
                    if pending:
                        started=time.monotonic()
                        results=annotate_group([(row,whole) for row,whole,_ in pending],processor,llm,
                            make_params,args.output,args.batch_size,args.batch_frame_budget,save,sha(CONFIG))
                        elapsed=time.monotonic()-started
                        decode_metrics={row['sample_id']:metrics for row,_,metrics in pending}
                        for result in results:
                            result['performance'].update(decode_metrics[result['sample_id']],
                                prefetch_wait_s=wait/len(results),process_s=elapsed/len(results),execution=execution)
                            save(args.output/'results'/f"{result['sample_id']}.json",result)
                        print('group complete',len(results),'process_s',elapsed,'wait_s',wait,flush=True)
                del loaded,pending
                tick=time.monotonic()
        return
    with closing(prefetched(assigned,load,args.prefetch)) as ready:
        for row,(whole,decode_metrics) in ready:
            wait=time.monotonic()-tick
            sid=row['sample_id'];target=args.output/'results'/f'{sid}.json'
            with lock(args.output,'sample-'+sid):
                if not target.exists():
                    started=time.monotonic()
                    result=annotate_sample(args,processor,llm,row,whole,make_params)
                    result['performance'].update(decode_metrics,prefetch_wait_s=wait,
                        process_s=time.monotonic()-started,execution=execution)
                    save(target,result)
                    print(sid,'complete',json.dumps(result['performance']),flush=True)
            del whole
            tick=time.monotonic()

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('vlm','export'):
        a=sub.add_parser(name);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='vlm':
            a.add_argument('--rank',type=int,default=0);a.add_argument('--shards',type=int,default=1);a.add_argument('--memory',type=float,default=.2)
            a.add_argument('--batch-size',type=int,default=4,help='Simultaneous windows; 1 for serial baseline')
            a.add_argument('--batch-frame-budget',type=int,default=256,help='Frame estimate per submission; never truncates a window')
            a.add_argument('--prefetch',type=int,choices=(0,1,2),default=1,help='Future clips, one background decoder per worker')
            a.add_argument('--scheduling',choices=('balanced','round-robin'),default='balanced')
            a.add_argument('--max-num-batched-tokens',type=int,default=4096)
            a.add_argument('--clip-batch-size',type=int,default=1,help='Clips sharing each window queue')
            a.add_argument('--decode-slots',type=int,default=1,help='Maximum simultaneous source decodes per host/output')
            a.add_argument('--decode-workers',type=int,default=1,help='CPU decode threads per prefetched clip group')
            a.add_argument('--prepared-media-cache',type=Path,help='Trusted node-local exact decoded-frame cache')
        else:
            a.add_argument('--vlm-results',type=Path,required=True);a.add_argument('--nod-results',type=Path,required=True)
    args=p.parse_args();{'vlm':vlm,'export':export}[args.command](args)

if __name__=='__main__':main()
