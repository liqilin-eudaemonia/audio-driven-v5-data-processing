"""Bounded request submission with a strictly single-sequence Omni engine.

Inputs, prompt, decoding and per-sample retry records retain their identities.
Distinct paired UUIDs prevent silent audio cache hits being paired with new video.
"""
import hashlib,time,queue,threading
from pathlib import Path
from .j_annotation import read,write,result_path


def groups(stream,limit,stop):
    if type(limit) is not int or limit not in (1,4):raise ValueError('Unvalidated request group')
    batch=[]
    for item in stream:
        if stop.is_set():break
        batch.append(item)
        if len(batch)==limit:
            yield batch;batch=[]
    if batch and not stop.is_set():yield batch


def ready_groups(stream,limit,stop):
    """Submit up to4 already-ready inputs; never wait just to fill a group.

    A bounded reader keeps CPU prefetch progressing during generation. Joining
    it precedes closing the source stream, so decoder/source locks remain owned.
    """
    if type(limit) is not int or not 1<=limit<=4:raise ValueError('Invalid ready group capacity')
    mailbox=queue.Queue(maxsize=limit);done=object();cancel=threading.Event()
    def put(value):
        while not cancel.is_set() and not stop.is_set():
            try:mailbox.put(value,timeout=.1);return
            except queue.Full:pass
    def reader():
        try:
            for item in stream:
                if cancel.is_set() or stop.is_set():break
                put(('item',item))
        except BaseException as exc:put(('error',exc))
        finally:put(('done',done))
    thread=threading.Thread(target=reader,name='omni-ready-submissions',daemon=True);thread.start()
    try:
        ended=False
        while not ended and not stop.is_set():
            try:kind,value=mailbox.get(timeout=.1)
            except queue.Empty:continue
            if kind=='error':raise value
            if kind=='done':break
            batch=[value];error=None
            while len(batch)<limit:
                try:kind,value=mailbox.get_nowait()
                except queue.Empty:break
                if kind=='done':ended=True;break
                if kind=='error':error=value;break
                batch.append(value)
            yield batch
            if error is not None:raise error
    finally:
        cancel.set()
        if hasattr(stream,'cancel_pending'):stream.cancel_pending()
        thread.join()


def request(processor,prepared,tag=None):
    ev=prepared['evidence'];ev['second_per_grid_ts']=[float(processor.image_processor.temporal_patch_size)/8]
    rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prepared['prompt']),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    out=dict(prompt=rendered,multi_modal_data=dict(video=[prepared['frames']],audio=[prepared['audio']]),mm_processor_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=ev['second_per_grid_ts']))
    if tag is not None:out['multi_modal_uuids']=dict(audio=[tag],video=[tag])
    return out


def generate(llm,processor,records,cfg,attempt_root,*,paired_ids=True,sampling_factory=None):
    """Return attempts in input order; retry only failed items, including resumes."""
    from .j_omni import fingerprint,immutable,generation_failures
    if sampling_factory is None:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        sampling_factory=lambda dec:SamplingParams(**dec,structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][^\n\r]*'))
    if len({r['row']['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate sample in request group')
    attempts=[[] for _ in records];pending=list(range(len(records)));run=str(time.time_ns())
    # Establish identical evidence before checking cached attempt fingerprints.
    requests=[request(processor,r['prepared']) for r in records]
    fingerprints=[fingerprint([r['prepared']['prompt'],r['prepared']['evidence']]) for r in records]
    for index,dec in enumerate(cfg['decoding_attempts']):
        paths={i:Path(attempt_root)/hashlib.sha256(records[i]['row']['sample_id'].encode()).hexdigest()[:2]/records[i]['row']['sample_id']/f'{index+1}.json' for i in pending}
        needed=[];values={}
        for i in pending:
            if paths[i].exists():
                att=read(paths[i])
                if att['input_fingerprint']!=fingerprints[i]:raise ValueError('Retry input changed')
                values[i]=att
            else:needed.append(i)
        if needed:
            submitted=[request(processor,records[i]['prepared'],f'{run}-{index}-{i}' if paired_ids else None) for i in needed]
            llm.reset_mm_cache()
            outputs=llm.generate(submitted,sampling_factory(dec),use_tqdm=False)
            if len(outputs)!=len(needed):raise ValueError('Inference output count differs')
            for i,req,out in zip(needed,submitted,outputs):
                if out.prompt is not None and out.prompt!=req['prompt']:raise ValueError('Inference output prompt differs')
                item=out.outputs[0]
                att=dict(caption=item.text,finish_reason=item.finish_reason,completion_tokens=len(item.token_ids),prompt_tokens=len(out.prompt_token_ids),prompt_tokens_sha256=fingerprint(out.prompt_token_ids),decoding=dec,failures=generation_failures(item.text,item.finish_reason),input_fingerprint=fingerprints[i])
                immutable(paths[i],att);values[i]=att
        retry=[]
        for i in pending:
            attempts[i].append(values[i])
            if values[i]['failures']:retry.append(i)
        pending=retry
        if not pending:break
    return [dict(attempts=a,rendered_prompt=req['prompt']) for a,req in zip(attempts,requests)]


def consume_source(stream,llm,processor,cfg,work,node,gpu,profile,revision,quality,state,status,stop,max_clips=None):
    from .j_omni import EvidenceUnavailable,immutable
    size=state['production_batch_size']
    adaptive=bool(cfg.get('adaptive_request_groups',{}).get(f'{node}-gpu-{gpu}',False))
    state['adaptive_request_groups']=adaptive
    group_iterator=(ready_groups if adaptive else groups)(stream,size,stop)
    try:
        for batch in group_iterator:
            records=[]
            for row,future,wait_s in batch:
                sid=row['sample_id']
                state['sample_id']=sid
                try:
                    tick=time.monotonic();prepared=future.result();wait_s+=time.monotonic()-tick;state['preparation_wait_s']+=wait_s
                    if profile['backend']=='process' and prepared['_preparation_metrics']['cuda_visible_devices']!='':raise ValueError('CPU decoder exposed CUDA')
                except EvidenceUnavailable as exc:
                    write(result_path(work,'held',sid),dict(sample_id=sid,source=row['source'],reason=repr(exc),at=time.time(),training_admitted=False));state['held_clips']+=1
                    if hasattr(stream,'persisted'):stream.persisted(sid)
                    continue
                except Exception as exc:
                    write(result_path(work,'errors',sid),dict(sample_id=sid,error=repr(exc),at=time.time()));raise
                scheduling=stream.scheduling_for(sid) if hasattr(stream,'scheduling_for') else None
                records.append(dict(row=row,prepared=prepared,wait_s=wait_s,scheduling=scheduling))
            if not records or stop.is_set():continue
            state.update(stage='inference',sample_id=records[0]['row']['sample_id'],job=records[0]['row']['source'].get('pair',state.get('job')),submitted_requests=len(records));status();tick=time.monotonic()
            from .j_omni_crop import attempts_root,result_sources
            generated=generate(llm,processor,records,cfg,attempts_root(work,cfg))
            elapsed=time.monotonic()-tick
            for rec,gen in zip(records,generated):
                row=rec['row'];sid=row['sample_id'];p=rec['prepared'];att=gen['attempts'][-1]
                result=dict(sample_id=sid,source=row['source'],pipeline_id=cfg['pipeline_id'],prompt=p['prompt'],rendered_prompt=gen['rendered_prompt'],caption=att['caption'],finish_reason=att['finish_reason'],generation_checks_passed=not att['failures'],review_flags=list(att['failures']),completion_tokens=att['completion_tokens'],attempts=gen['attempts'],input_evidence=p['evidence'],audio_recipe=p['recipe'],seconds=elapsed,seconds_scope='shared_request_group',node=node,gpu=gpu,execution_revision=revision,engine_max_num_seqs=1,production_batch_size=size,adaptive_request_groups=adaptive,submitted_requests=len(records),preparation_profile=profile,preparation_metrics=dict(p['_preparation_metrics'],main_wait_s=rec['wait_s']),completed_at=time.time(),training_admitted=False,human_ground_truth=False)
                if quality is not None:result['visual_quality']=quality[sid]
                if rec.get('scheduling') is not None:result['scheduling']=rec['scheduling']
                result.update(result_sources(row,p))
                immutable(result_path(work,'caption',sid),result)
                if hasattr(stream,'persisted'):stream.persisted(sid)
                state['completed_clips']+=1;state['failed_clips']+=bool(att['failures']);state['last_progress_at']=time.time();status()
            state['stage']='preparing'
            if max_clips and state['completed_clips']>=max_clips:stop.set()
    except BaseException as exc:
        if hasattr(stream,'record_failure'):stream.record_failure(state.get('sample_id'),exc)
        raise
    finally:
        group_iterator.close()
