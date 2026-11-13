"""Streaming J-pair Omni captioning from committed nod evidence and official audio masks.

Shared work queue uses filesystem locks, never remote PID guesses. No WAV exports.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tarfile
import threading
import time
import traceback

from listen_benchmark.j_annotation import read, write, acquire, locked, JobQueue, result_path
from listen_benchmark.native_caption_nod import render, PROMPT_STYLES
from listen_benchmark.native_caption_quality import generation_failures
from listen_benchmark.j_omni_prefetch import PreparationPool

STYLE = 'first_plus_evidence'
MODEL = '/private/ckpts/Qwen3-Omni/Instruct'
SCHEMA = 'j-listener-segment-official-union-v1'
EXECUTION_REVISION = 'paired-av-cache-reset-v2+pts-tail-validation-v1'


class EvidenceUnavailable(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


@lru_cache(maxsize=64)
def checked_file(path,size,mtime,expected):
    if file_sha(path)!=expected:raise ValueError('Source checksum mismatch: '+path)
    return expected


def check_file(path,expected):
    st=Path(path).stat();return checked_file(str(path),st.st_size,st.st_mtime_ns,expected)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable record differs: '+str(path))
    else:write(path,value)


def verified_job(path):
    job=read(path)
    if not job['rows']:raise ValueError('Empty source job')
    src=job['rows'][0]['source'];check_file(src['source_result'],job['source_result_sha256'])
    if src['pair']!=job['job_id']:raise ValueError('Job identity differs')
    return job


def official_metadata(source,work):
    member=str(Path(source['member']).with_suffix('.json'))
    archive=Path(source['archive']);st=archive.stat()
    key=fingerprint([str(archive),member,st.st_size,st.st_mtime_ns])
    path=Path(work)/'official_metadata'/key[:2]/(key+'.json')
    with locked(str(path)+'.lock'):
        if path.exists():out=read(path)
        else:
            with tarfile.open(archive,'r:') as tar:
                entry=tar.getmember(member)
                if not entry.isfile():raise ValueError('Metadata is not a regular file')
                raw=tar.extractfile(entry).read()
            after=archive.stat()
            if (after.st_size,after.st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise ValueError('Archive changed')
            out={'raw':json.loads(raw),'provenance':dict(archive=str(archive),member=member,
                 sha256=hashlib.sha256(raw).hexdigest(),archive_size=st.st_size,archive_mtime_ns=st.st_mtime_ns)}
            write(path,out)
    for k in ('metadata:transcript','metadata:vad'):
        if not isinstance(out['raw'].get(k),list):raise ValueError('Missing official metadata is not silence: '+k)
    return out


def audio_recipe(row,work):
    import soundfile as sf
    from listen_audio_postprocess.bulk import official
    from listen_benchmark.official_audio_gate import POLICY
    s=row['source'];a=s['normalized_audio'];sr=a['sample_rate']
    if s['sample_id']!=row['sample_id'] or Path(s['source']['member']).stem!=s['listener_id']:
        raise ValueError('Listener identity mismatch')
    check_file(a['path'],a['sha256']);info=sf.info(a['path'])
    first,last=round(s['source_start_s']*sr),round(s['source_end_s']*sr)
    if info.samplerate!=sr or info.channels!=1 or not 0<=first<last<=info.frames:
        raise ValueError('Invalid source audio range')
    if abs((last-first)/sr-s['duration_s'])>1e-8:raise ValueError('Audio duration differs from pair')
    raw=official_metadata(s['source'],work)
    off=official(raw['raw'],first/sr,(last-first)/sr)
    recipe=dict(schema=SCHEMA,sample_id=row['sample_id'],listener_id=s['listener_id'],
                source=dict(a,channels=1,source_samples=info.frames,first_sample=first,last_sample=last),
                official_listener={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in off[k]] for k in ('words','sentences','vad')},
                gate_policy=POLICY,extra_mute_intervals=[],language_filter='disabled_user_official_union_only',
                provenance=raw['provenance'],source_time_origin_s=first/sr)
    immutable(result_path(work,'audio_recipes',row['sample_id']),recipe)
    return recipe


def read_audio(recipe):
    import numpy as np
    import soundfile as sf
    from listen_benchmark.official_audio_gate import make_gate,POLICY
    from listen_audio_postprocess.recipes import waveform_digest
    if recipe['schema']!=SCHEMA or recipe['gate_policy']!=POLICY or recipe['extra_mute_intervals']:
        raise ValueError('Unexpected audio processing policy')
    s=recipe['source'];check_file(s['path'],s['sha256'])
    with sf.SoundFile(s['path']) as f:
        if f.samplerate!=s['sample_rate'] or f.channels!=1 or len(f)!=s['source_samples']:raise ValueError('Audio format changed')
        f.seek(s['first_sample']);wave=f.read(s['last_sample']-s['first_sample'],dtype='float32',always_2d=True)
    if len(wave)!=s['last_sample']-s['first_sample'] or not np.isfinite(wave).all():raise ValueError('Incomplete audio')
    gate,info=make_gate(recipe['official_listener'],len(wave),s['sample_rate'])
    processed=wave*gate[:,None]
    assert np.all(processed[gate==0]==0) and processed.shape==wave.shape
    return processed,s['sample_rate'],dict(recipe_sha256=fingerprint(recipe),
        processed_waveform_sha256=waveform_digest(processed,s['sample_rate']),
        source_segment_waveform_sha256=waveform_digest(wave,s['sample_rate']),
        full_silence=not bool(np.any(processed)),positive_samples=info['positive_samples'],
        source_start_sample=s['first_sample'],source_end_sample=s['last_sample'])


def nod_evidence(row,old_work):
    path=result_path(old_work,'nod',row['sample_id'])
    if not path.exists():raise FileNotFoundError('Nod detection has not finished: '+str(path))
    n=read(path)
    if n['sample_id']!=row['sample_id'] or n['source']!=row['source'] or n['fps']!=25 or n['profile']!='recall':
        raise ValueError('Nod identity/crop/time/profile mismatch')
    if abs(n['duration_s']-row['source']['duration_s'])>1e-8:raise ValueError('Nod duration mismatch')
    if n['status']!='ok':raise EvidenceUnavailable('Nod evidence unavailable; not an empty detection: '+str(n.get('reason')))
    if not isinstance(n['events'],list):raise ValueError('Invalid nod events')
    return dict(sample_id=row['sample_id'],fps=25,profile='recall',
        events=[dict(id=e['event_id'],start=e['start'],end=e['end']) for e in n['events']],
        detector_file=str(path),detector_sha256=file_sha(path))


def prepare_input(row,old_work,work,resource_work):
    import numpy as np
    from scipy.signal import resample_poly
    from listen_benchmark.full_raw import prepare
    from listen_benchmark.media import uniform_video
    from listen_visual_native8_nod import decode_slot
    cfg=read(Path(work)/'config.json')
    nod=nod_evidence(row,old_work);recipe=audio_recipe(row,work);wave,sr,identity=read_audio(recipe)
    # Host-local decoding budget shared with that host's nod workers, even when a
    # caption worker claims a source originally owned by the other host.
    with decode_slot(Path(resource_work)/'resources',cfg['decode_slots']):
        media=prepare(row['source'],'visual')
    if abs(media['meta']['video']['source_time_origin_s']-recipe['source_time_origin_s'])>1e-9:
        raise ValueError('Audio/video origins differ')
    frames,evidence=uniform_video(media,8)
    # Same first-frame-at-or-after sampling, bicubic crop, JPEG85 pixels as existing J visual path.
    audio=wave[:,0]
    if sr!=16000:
        g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g).astype(np.float32)
    prompt_row=dict(sample_id=row['sample_id'],duration_s=row['source']['duration_s'],nod_supplement=nod,
                    audio_preflight=dict(full_silence=identity['full_silence']))
    prompt=render(prompt_row,style=STYLE)
    evidence.update(audio_identity=identity,listener_audio_is_silent=identity['full_silence'],
        audio_encoder_rate=16000,audio_encoder_samples=len(audio),audio_origin='virtual_official_listener_union',
        video_evidence=media['meta']['video'],nod_supplement=nod,video_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        audio_encoder_sha256=hashlib.sha256(audio.tobytes()).hexdigest())
    return dict(frames=frames,audio=audio,prompt=prompt,evidence=evidence,recipe=recipe)


def producer_step(work,limit=128):
    from listen_benchmark.j_omni_sources import source_nodes,journal_for,resolve_nod_receipt,validate_item
    work=Path(work);cfg=read(work/'config.json')
    with locked(work/'producer-step.lock'):
        state=read(work/'producer_state.json') if (work/'producer_state.json').exists() else {'nodes':{},'published_sources':0,'published_clips':0}
        queue=JobQueue(work,'caption');q=read(queue.path)
        if state['published_sources']-q['completed']>cfg.get('backlog_sources',1024):
            return dict(state,stage='backpressure',at=time.time())
        sources=source_nodes(cfg)
        for name,node in sources.items():
            if not node.get('old_work'):continue  # compute-only node, no new source partition
            old=Path(node['old_work']);s=state['nodes'].setdefault(name,dict(offset=0,pending={},scanned_sources=0))
            nq=read(old/'queues/nod/state.json')
            journal,ready_offset=journal_for(node,nq)
            skip_offsets=set(node.get('skip_duplicate_offsets',[]))
            with journal.open() as f:
                f.seek(s['offset'])
                for _ in range(limit):
                    position=f.tell()
                    if position>=ready_offset:break
                    line=f.readline()
                    if not line or not line.endswith('\n'):break
                    s['offset']=f.tell();s['scanned_sources']+=1
                    if position in skip_offsets:continue
                    item=json.loads(line);validate_item(node,item);s['pending'][item['job_id']]=item
            for jid,item in list(s['pending'].items()):
                validate_item(node,item)
                resolved=resolve_nod_receipt(cfg,old,item)
                if resolved is None:continue
                nod_owner,done,receipt=resolved
                jobpath=work/'jobs'/f'{jid}.json';jobpath.parent.mkdir(parents=True,exist_ok=True)
                original=old/'jobs'/f'{jid}.json'
                if jobpath.is_symlink():
                    if jobpath.resolve()!=original.resolve():raise ValueError('Duplicate source across nodes')
                elif jobpath.exists():raise ValueError('Unexpected job path')
                else:jobpath.symlink_to(original)
                immutable(work/'job_origins'/f'{jid}.json',dict(node=name,old_work=str(nod_owner),nod_receipt=str(done),source_result_sha256=receipt['source_result_sha256']))
                # Journal append precedes cursor commit: a crash can duplicate a line,
                # but JobQueue's claims/completion receipts prevent duplicate work.
                with (work/'ready.jsonl').open('a') as f:
                    f.write(json.dumps(item)+'\n');f.flush();os.fsync(f.fileno())
                state['published_sources']+=1;state['published_clips']+=item['count'];del s['pending'][jid]
            s['upstream_failed_sources']=list(nq.get('failed',{}))
        state.update(at=time.time(),stage='feeding')
        state['all_sources_published']=all(n in state['nodes'] and state['nodes'][n]['scanned_sources']==node['scope_sources'] and not state['nodes'][n]['pending'] for n,node in sources.items() if node.get('old_work'))
        write(work/'producer_state.json',state)
        if state['all_sources_published']:write(work/'producer_complete.json',state)
        return state


def producer(work):
    work=Path(work);owner=acquire(work/'producer.lock',False)
    if owner is None:return
    stopped=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    while not stopped.is_set():
        try:
            s=producer_step(work);write(work/'producer.json',dict(s,pid=os.getpid(),error=None))
            if s.get('all_sources_published'):break
        except Exception as e:write(work/'producer.json',dict(pid=os.getpid(),at=time.time(),error=repr(e)))
        stopped.wait(2)
    owner.close()


def pending_caption_rows(work,job,pipeline_id,quality=None):
    """Reject/review rows never enter the media preparation or model stream."""
    todo=[];added={'exclude':0,'review':0}
    for row in sorted(job['rows'],key=lambda r:-r['source']['duration_s']):
        sid=row['sample_id'];dest=result_path(work,'caption',sid)
        if dest.exists():
            old=read(dest)
            if old['source']!=row['source'] or old['pipeline_id']!=pipeline_id:raise ValueError('Existing output provenance mismatch')
        elif result_path(work,'held',sid).exists():
            if read(result_path(work,'held',sid))['source']!=row['source']:raise ValueError('Held source mismatch')
        elif quality is not None and quality[sid]['decision']!='retain':
            d=quality[sid];branch='visual_excluded' if d['decision']=='exclude' else 'visual_review'
            path=result_path(work,branch,sid);fresh=not path.exists()
            immutable(path,dict(sample_id=sid,source=row['source'],visual_quality=d,visual_gate_applied=True,training_admitted=False))
            added[d['decision']]+=fresh
        else:todo.append(row)
    return todo,added


def worker(work,node,gpu,max_clips=None):
    import numpy as np
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    work=Path(work);cfg=read(work/'config.json');name=f'{node}-gpu-{gpu}'
    baseline=dict(backend='thread',workers=2,prefetch=2,ready_first=False)
    profile=cfg.get('preparation_profiles',{}).get(name,baseline)
    if set(profile)!=set(baseline) or type(profile['ready_first']) is not bool or type(profile['prefetch']) is not int or not 1<=profile['prefetch']<=32:raise ValueError('Invalid preparation profile')
    revision=EXECUTION_REVISION if profile==baseline else EXECUTION_REVISION+'+process-prefetch-v1' 
    quality_gate=None
    if cfg.get('visual_quality_gate'):
        from listen_benchmark.j_visual_gate import QualityGate
        quality_gate=QualityGate(cfg['visual_quality_gate']);revision+='+visual-gate-v1'
    engine_seqs=cfg.get('engine_max_num_seqs',{}).get(name,1)
    if type(engine_seqs) is not int or engine_seqs not in (1,4):raise ValueError('Unvalidated engine capacity')
    inference_benchmark=cfg.get('startup_inference_benchmarks',{}).get(name)
    if engine_seqs!=1:revision+='+batch-capacity-canary-v1'
    submission_size=cfg.get('queued_requests_per_call',{}).get(name,1)
    if type(submission_size) is not int or submission_size not in (1,4) or (submission_size>1 and engine_seqs!=1):raise ValueError('Unvalidated queued request configuration')
    queue_benchmark=cfg.get('startup_queue_benchmarks',{}).get(name)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    if os.uname().nodename!=cfg['nodes'][node]['hostname']:raise ValueError('Wrong host')
    owner=acquire(work/'worker_locks'/f'{name}.lock',False)
    if owner is None:raise RuntimeError('Worker already running')
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    state=dict(pid=os.getpid(),node=node,gpu=gpu,stage='loading_model',started_at=time.time(),completed_clips=0,failed_clips=0,held_clips=0,visual_excluded_clips=0,visual_review_clips=0,execution_revision=revision,preparation_profile=profile,preparation_wait_s=0.,engine_max_num_seqs=engine_seqs,production_batch_size=1)
    def status():write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
    def heartbeat():
        while not stop.wait(10):status()
    status();threading.Thread(target=heartbeat,daemon=True).start()
    for rel,h in cfg['code_hashes'].items():
        if file_sha(Path(cfg['project_root'])/rel)!=h:raise ValueError('Runtime code changed: '+rel)
    processor=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
    try:
        llm=LLM(model=MODEL,trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=cfg['nodes'][node].get('memory',.85),
            max_model_len=49152,max_num_seqs=engine_seqs,max_num_batched_tokens=49152,enable_chunked_prefill=False,
            limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,enforce_eager=False,
            compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=([1] if engine_seqs==1 else [1,2,4]),max_cudagraph_capture_size=engine_seqs))
    except Exception as exc:
        if inference_benchmark:
            from listen_benchmark.j_omni_batch_canary import rollback
            rollback(work,name,inference_benchmark,exc)
        raise
    benchmark=cfg.get('startup_benchmarks',{}).get(name)
    if benchmark:
        from listen_benchmark.j_omni_e2e import run as benchmark_run
        state['stage']='execution_benchmark';status()
        def progress(phase,count,sid):
            state.update(benchmark_phase=phase,benchmark_clips=count,sample_id=sid,last_progress_at=time.time());status()
        benchmark_result=benchmark_run(llm,processor,cfg,work,node,gpu,benchmark,stop,progress)
        if not benchmark_result or not benchmark_result['safe_speed_gain']:
            profile=cfg.get('benchmark_fallback_profiles',{}).get(name,baseline)
            revision=EXECUTION_REVISION if profile==baseline else EXECUTION_REVISION+'+process-prefetch-v1'
            state.update(preparation_profile=profile,execution_revision=revision,benchmark_fallback_to_baseline=True)

    if inference_benchmark:
        from listen_benchmark.j_omni_batch_canary import run as inference_run,rollback
        state['stage']='execution_inference_benchmark';status()
        def inference_progress(phase,count,sid):
            state.update(benchmark_phase=phase,benchmark_clips=count,sample_id=sid,last_progress_at=time.time());status()
        try:
            inference_result=inference_run(llm,processor,cfg,work,node,gpu,inference_benchmark,stop,inference_progress)
            state['inference_benchmark_safe_gain']=bool(inference_result and inference_result['safe_speed_gain'])
        except Exception as exc:
            state['last_error']=repr(exc);status();rollback(work,name,inference_benchmark,exc);raise

    if queue_benchmark:
        from listen_benchmark.j_omni_queued_e2e import run as queue_run
        state['stage']='execution_queue_benchmark';status()
        def queue_progress(phase,count,sid):
            state.update(benchmark_phase=phase,benchmark_clips=count,sample_id=sid,last_progress_at=time.time());status()
        try:
            result=queue_run(llm,processor,cfg,work,node,gpu,queue_benchmark,stop,queue_progress)
            if not result or not result['safe_speed_gain']:submission_size=1
            state['queue_benchmark_safe_gain']=bool(result and result['safe_speed_gain'])
        except Exception as exc:
            write(Path(queue_benchmark)/'error.json',dict(at=time.time(),error=repr(exc),production_fallback=1))
            submission_size=1;state['queue_benchmark_error']=repr(exc)
    if submission_size>1:revision+='+queued4-single-sequence-v1'
    state.update(execution_revision=revision,production_batch_size=submission_size);status()

    recovery_jobs=list(cfg.get('startup_recovery_jobs',{}).get(name,[]))
    q=JobQueue(work,'caption');state['stage']='waiting';status()
    with PreparationPool(profile['backend'],profile['workers']) as pool:
        while not stop.is_set():
            if recovery_jobs:
                claim=q.claim_failed(recovery_jobs.pop(0),name)
                if claim is None:continue
            else:
                claim=q.claim(name,eligible=quality_gate.ready if quality_gate else None)
            if claim is None:
                if (work/'producer_complete.json').exists():
                    qs=read(q.path)
                    if not qs['inflight'] and not qs.get('deferred'):break
                state['stage']='waiting';status();stop.wait(3);continue
            jid,handle=claim;state.update(stage='preparing',job=jid);status()
            try:
                origin=read(work/'job_origins'/f'{jid}.json');job=verified_job(work/'jobs'/f'{jid}.json')
                if job['source_result_sha256']!=origin['source_result_sha256']:raise ValueError('Completion source differs')
                quality=quality_gate.decisions(job) if quality_gate else None
                rows=job['rows'];todo,added=pending_caption_rows(work,job,cfg['pipeline_id'],quality)
                state['visual_excluded_clips']+=added['exclude'];state['visual_review_clips']+=added['review']
                if quality_gate:state.update(visual_quality_contract=quality_gate.contract,last_progress_at=time.time())
                stream=pool.stream(todo,origin['old_work'],work,cfg['nodes'][node]['resource_work'],profile['prefetch'],stop,profile['ready_first'])
                try:
                    if submission_size>1:
                        from listen_benchmark.j_omni_queued import consume_source
                        consume_source(stream,llm,processor,cfg,work,node,gpu,profile,revision,quality,state,status,stop,max_clips)
                    else:
                        for row,future,wait_s in stream:
                            if stop.is_set():break
                            sid=row['sample_id'];dest=result_path(work,'caption',sid)
                            try:
                                wait_start=time.monotonic();prepared=future.result();wait_s+=time.monotonic()-wait_start
                                state['preparation_wait_s']+=wait_s
                                if profile['backend']=='process' and prepared['_preparation_metrics']['cuda_visible_devices']!='':raise ValueError('CPU decoder exposed CUDA')
                            except EvidenceUnavailable as exc:
                                # Missing/unavailable evidence is quarantined, never rendered as empty cues.
                                write(result_path(work,'held',sid),dict(sample_id=sid,source=row['source'],reason=repr(exc),at=time.time(),training_admitted=False))
                                state['held_clips']+=1;continue
                            except Exception as exc:
                                write(result_path(work,'errors',sid),dict(sample_id=sid,error=repr(exc),at=time.time()))
                                raise
                            ev=prepared['evidence'];ev['second_per_grid_ts']=[float(processor.image_processor.temporal_patch_size)/8]
                            rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prepared['prompt']),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                            request=dict(prompt=rendered,multi_modal_data=dict(video=[prepared['frames']],audio=[prepared['audio']]),
                                mm_processor_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=ev['second_per_grid_ts']))
                            state.update(stage='inference',sample_id=sid);status();attempts=[];start=time.monotonic()
                            for index,dec in enumerate(cfg['decoding_attempts']):
                                path=work/'attempts'/hashlib.sha256(sid.encode()).hexdigest()[:2]/sid/f'{index+1}.json'
                                if path.exists():
                                    att=read(path)
                                    if att['input_fingerprint']!=fingerprint([prepared['prompt'],ev]):raise ValueError('Retry input changed')
                                else:
                                    # Cache hits are per modality; repeated silent audio
                                    # with a new video otherwise sends unpaired video to
                                    # the installed Omni processor. Its cache-disabled
                                    # path also miscounts audio placeholders. Retain the
                                    # paired cache-miss path and clear both endpoints
                                    # between synchronous requests via vLLM's public API.
                                    llm.reset_mm_cache()
                                    out=llm.generate([request],SamplingParams(**dec,structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][^\n\r]*')),use_tqdm=False)[0]
                                    item=out.outputs[0];att=dict(caption=item.text,finish_reason=item.finish_reason,
                                        completion_tokens=len(item.token_ids),prompt_tokens=len(out.prompt_token_ids),decoding=dec,
                                        failures=generation_failures(item.text,item.finish_reason),input_fingerprint=fingerprint([prepared['prompt'],ev]))
                                    immutable(path,att)
                                attempts.append(att)
                                if not att['failures']:break
                            result=dict(sample_id=sid,source=row['source'],pipeline_id=cfg['pipeline_id'],prompt=prepared['prompt'],rendered_prompt=rendered,
                                caption=att['caption'],finish_reason=att['finish_reason'],generation_checks_passed=not att['failures'],
                                review_flags=list(att['failures']),completion_tokens=att['completion_tokens'],attempts=attempts,
                                input_evidence=ev,audio_recipe=prepared['recipe'],seconds=time.monotonic()-start,
                                node=node,gpu=gpu,execution_revision=revision,engine_max_num_seqs=engine_seqs,production_batch_size=1,preparation_profile=profile,preparation_metrics=dict(prepared['_preparation_metrics'],main_wait_s=wait_s),completed_at=time.time(),training_admitted=False,human_ground_truth=False)
                            if quality is not None:result['visual_quality']=quality[sid]
                            immutable(dest,result);state['completed_clips']+=1;state['failed_clips']+=bool(att['failures']);state['last_progress_at']=time.time();status()
                            state['stage']='preparing'
                            if max_clips and state['completed_clips']>=max_clips:stop.set()
                finally:
                    stream.close()  # drain/cancel preparation before source claim release
                completed=sum(result_path(work,'caption',r['sample_id']).exists() for r in rows)
                held=sum(result_path(work,'held',r['sample_id']).exists() for r in rows)
                excluded=sum(result_path(work,'visual_excluded',r['sample_id']).exists() for r in rows)
                review=sum(result_path(work,'visual_review',r['sample_id']).exists() for r in rows)
                if completed+held+excluded+review==len(rows):
                    bad=sum(not read(result_path(work,'caption',r['sample_id']))['generation_checks_passed'] for r in rows if result_path(work,'caption',r['sample_id']).exists())
                    receipt=dict(job_id=jid,clips=len(rows),caption_clips=completed,failed_generation_clips=bad,held_clips=held,visual_excluded_clips=excluded,visual_review_clips=review,completed_at=time.time())
                    if quality_gate:receipt['visual_quality_contract']=quality_gate.contract
                    q.finish(jid,receipt=receipt);state['last_progress_at']=time.time();status()
                elif not stop.is_set():q.finish(jid,error='Missing caption outputs')
            except Exception as exc:
                traceback.print_exc()
                state['last_error']=repr(exc);q.finish(jid,error=repr(exc));stop.wait(2)
            finally:handle.close()
    state['stage']='stopped' if stop.is_set() else 'complete';status();stop.set();owner.close()
