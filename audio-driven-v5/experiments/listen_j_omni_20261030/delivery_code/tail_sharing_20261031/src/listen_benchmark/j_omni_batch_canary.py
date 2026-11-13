"""Bounded inference-only batch experiment; never enables production batching."""
import hashlib
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .j_annotation import read,write,locked
from .j_omni_prefetch import cpu_only


def rollback(work,name,folder,error):
    """A failed experiment must not repeatedly reload its failing engine settings."""
    work=Path(work)
    with locked(work/'config-edit.lock'):
        cfg=read(work/'config.json')
        cfg.get('engine_max_num_seqs',{}).pop(name,None)
        cfg.get('startup_inference_benchmarks',{}).pop(name,None)
        write(work/'config.json',cfg)
    write(Path(folder)/'error.json',dict(at=time.time(),worker=name,error=repr(error),engine_reverted_to_single_request=True))


def request(processor,prepared,tag=None):
    rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prepared['prompt']),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    result=dict(prompt=rendered,multi_modal_data=dict(video=[prepared['frames']],audio=[prepared['audio']]),mm_processor_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=[float(processor.image_processor.temporal_patch_size)/8]))
    if tag is not None:
        # Every request misses BOTH modality caches, including repeated silence.
        result['multi_modal_uuids']=dict(audio=[tag],video=[tag])
    return result


def run(llm,processor,cfg,work,node,gpu,folder,stop,progress):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from .j_omni import fingerprint,generation_failures
    from .j_omni_e2e import prepare_benchmark
    folder=Path(folder);tasks=read(folder/'tasks.json');reference=read(folder/'reference_captions.json')
    receipt=folder/'complete.json'
    if receipt.exists():
        result=read(receipt)
        if result['pipeline_id']!=cfg['pipeline_id'] or result['tasks_sha256']!=fingerprint(tasks):raise ValueError('Inference benchmark inputs changed')
        return result
    run_id=str(time.time_ns());prepared=[];prepare_started=time.monotonic()
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_only) as pool:
        futures=[pool.submit(prepare_benchmark,t['row'],t['old_work'],str(work),cfg['nodes'][node]['resource_work']) for t in tasks]
        for index,(task,future) in enumerate(zip(tasks,futures)):
            if stop.is_set():return
            value=future.result();expected=task['expected']
            if value['prompt']!=task['prompt']:raise ValueError('Prompt differs')
            for key,data in [('video_frames_sha256',value['frames']),('audio_encoder_sha256',value['audio'])]:
                if hashlib.sha256(data.tobytes()).hexdigest()!=expected[key] or value['evidence'][key]!=expected[key]:raise ValueError('Prepared media differs')
            prepared.append(value);progress('preparing',index+1,task['row']['sample_id'])
    prepare_seconds=time.monotonic()-prepare_started
    reports=[]
    for phase,batch_size in enumerate([1,4,4,1]):
        if stop.is_set():return
        llm.reset_prefix_cache();started=time.monotonic();records=[]
        for offset in range(0,len(tasks),batch_size):
            indices=list(range(offset,min(offset+batch_size,len(tasks))))
            attempts={i:[] for i in indices};pending=indices[:]
            for retry,dec in enumerate(cfg['decoding_attempts']):
                if stop.is_set():return
                requests=[request(processor,prepared[i],f'{run_id}-{phase}-{i}-{retry}' if batch_size>1 else None) for i in pending]
                llm.reset_mm_cache()
                outputs=llm.generate(requests,SamplingParams(**dec,structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][^\n\r]*')),use_tqdm=False)
                if len(outputs)!=len(pending):raise ValueError('Inference output count differs')
                retry_indices=[]
                for i,req,out in zip(pending,requests,outputs):
                    if out.prompt is not None and out.prompt!=req['prompt']:raise ValueError('Inference output prompt association differs')
                    item=out.outputs[0];failures=generation_failures(item.text,item.finish_reason)
                    attempts[i].append(dict(caption=item.text,finish_reason=item.finish_reason,failures=failures,completion_tokens=len(item.token_ids),prompt_tokens=len(out.prompt_token_ids),prompt_tokens_sha256=fingerprint(out.prompt_token_ids),request_id=out.request_id,paired_uuid=req.get('multi_modal_uuids'),decoding=dec))
                    if failures:retry_indices.append(i)
                pending=retry_indices
                if not pending:break
            for i in indices:
                final=attempts[i][-1];sid=tasks[i]['row']['sample_id']
                records.append(dict(sample_id=sid,attempts=attempts[i],caption=final['caption'],passed=not final['failures'],prompt_tokens_sha256=final['prompt_tokens_sha256'],matches_prior_reference=final['caption']==reference[sid]))
                progress(phase,len(records),sid)
        elapsed=time.monotonic()-started
        report=dict(phase=phase,batch_size=batch_size,seconds=elapsed,clips=len(records),clips_per_second=len(records)/elapsed,normal_outputs=sum(r['passed'] for r in records),outputs=records)
        write(folder/run_id/f'phase-{phase}.json',report);reports.append(report)
    indexed=[{r['sample_id']:r for r in p['outputs']} for p in reports];comparisons=[]
    for task in tasks:
        sid=task['row']['sample_id'];rows=[p[sid] for p in indexed]
        comparisons.append(dict(sample_id=sid,all_four_captions_identical=len({r['caption'] for r in rows})==1,all_prompt_tokens_identical=len({r['prompt_tokens_sha256'] for r in rows})==1,all_normal=all(r['passed'] for r in rows),baseline_matches_prior_reference=rows[0]['matches_prior_reference'] and rows[3]['matches_prior_reference']))
    base=reports[0]['seconds']+reports[3]['seconds'];batched=reports[1]['seconds']+reports[2]['seconds']
    result=dict(at=time.time(),pipeline_id=cfg['pipeline_id'],tasks_sha256=fingerprint(tasks),run_id=run_id,node=node,gpu=gpu,batch_sizes=[1,4,4,1],preparation_seconds_excluded=prepare_seconds,base_seconds=base,batch_seconds=batched,speedup=base/batched,normal_outputs=sum(p['normal_outputs'] for p in reports),total_outputs=sum(p['clips'] for p in reports),all_raw_media_and_prompts_identical=True,comparisons=comparisons,scope='Inference-only candidate; no production batching enabled or end-to-end speed claim')
    result['safe_speed_gain']=result['speedup']>=1.1 and all(all(r[k] for k in ['all_four_captions_identical','all_prompt_tokens_identical','all_normal','baseline_matches_prior_reference']) for r in comparisons)
    write(receipt,result)
    return result
