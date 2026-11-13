"""Same-model ABBA validation before a worker resumes its production queue."""
import hashlib,time
from pathlib import Path
from .j_annotation import read,write
from .j_omni_prefetch import PreparationPool
from .j_omni import fingerprint,generation_failures

def run(llm,processor,cfg,work,node,gpu,folder,stop,progress):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    folder=Path(folder);receipt=folder/'complete.json';tasks=read(folder/'tasks.json')
    if receipt.exists():
        r=read(receipt)
        if r['pipeline_id']!=cfg['pipeline_id'] or r['tasks_sha256']!=fingerprint(tasks):raise ValueError('Benchmark inputs differ')
        return r
    items=[x['row'] for x in tasks];lookup={x['row']['sample_id']:x for x in tasks}
    # prepare_call needs a source-specific nod root; use a top-level dispatch.
    from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
    from .j_omni_prefetch import cpu_only,prefetch_stream
    import multiprocessing
    run_id=str(time.time_ns());reports=[]
    profiles=[dict(backend='thread',workers=2,prefetch=2,ready_first=False),dict(backend='process',workers=4,prefetch=8,ready_first=True)]
    pools=[ThreadPoolExecutor(max_workers=2),ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_only)]
    try:
        for index,arm in enumerate([0,1,1,0]):
            if stop.is_set():return
            llm.reset_prefix_cache();profile=profiles[arm];started=time.monotonic();outputs=[]
            def submit(row):
                return pools[arm].submit(prepare_benchmark,row,lookup[row['sample_id']]['old_work'],str(work),cfg['nodes'][node]['resource_work'])
            stream=prefetch_stream(items,submit,profile['prefetch'],stop,profile['ready_first'])
            try:
                for row,future,wait_s in stream:
                    if stop.is_set():return
                    p=future.result();expected=lookup[row['sample_id']]['expected']
                    assert p['evidence']['video_frames_sha256']==expected['video_frames_sha256'] and p['evidence']['audio_encoder_sha256']==expected['audio_encoder_sha256']
                    assert hashlib.sha256(p['frames'].tobytes()).hexdigest()==expected['video_frames_sha256'] and hashlib.sha256(p['audio'].tobytes()).hexdigest()==expected['audio_encoder_sha256']
                    assert p['prompt']==lookup[row['sample_id']]['prompt']
                    rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=p['prompt']),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    request=dict(prompt=rendered,multi_modal_data=dict(video=[p['frames']],audio=[p['audio']]),mm_processor_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=[float(processor.image_processor.temporal_patch_size)/8]))
                    attempts=[];gen=time.monotonic()
                    for dec in cfg['decoding_attempts']:
                        llm.reset_mm_cache();out=llm.generate([request],SamplingParams(**dec,structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][^\n\r]*')),use_tqdm=False)[0];item=out.outputs[0]
                        attempts.append(dict(caption=item.text,finish_reason=item.finish_reason,failures=generation_failures(item.text,item.finish_reason),completion_tokens=len(item.token_ids),prompt_tokens=len(out.prompt_token_ids)))
                        if not attempts[-1]['failures']:break
                    outputs.append(dict(sample_id=row['sample_id'],attempts=attempts,caption=attempts[-1]['caption'],passed=not attempts[-1]['failures'],input_fingerprint=fingerprint([p['prompt'],p['evidence']['video_frames_sha256'],p['evidence']['audio_encoder_sha256']]),generation_seconds=time.monotonic()-gen,preparation=p['_preparation_metrics']))
                    progress(index,len(outputs),row['sample_id'])
            finally:stream.close()
            elapsed=time.monotonic()-started;report=dict(phase=index,profile=profile,clips=len(outputs),seconds=elapsed,clips_per_second=len(outputs)/elapsed,all_inputs_identical=True,normal_outputs=sum(x['passed'] for x in outputs),outputs=outputs)
            write(folder/run_id/f'phase-{index}.json',report);reports.append(report)
        assert all(r['clips']==len(items) for r in reports)
        base=sum(r['seconds'] for r in reports if r['profile']['backend']=='thread');fast=sum(r['seconds'] for r in reports if r['profile']['backend']=='process')
        byphase=[{r['sample_id']:r for r in p['outputs']} for p in reports];comparisons=[]
        for sid in lookup:
            assert len({p[sid]['input_fingerprint'] for p in byphase})==1
            comparisons.append(dict(sample_id=sid,all_four_captions_identical=len({p[sid]['caption'] for p in byphase})==1,all_normal=all(p[sid]['passed'] for p in byphase)))
        result=dict(at=time.time(),pipeline_id=cfg['pipeline_id'],tasks_sha256=fingerprint(tasks),run_id=run_id,node=node,gpu=gpu,profiles=profiles,clips_per_phase=len(items),phases=4,base_seconds=base,fast_seconds=fast,speedup=base/fast,relative_gain=base/fast-1,comparisons=comparisons,normal_outputs=sum(r['normal_outputs'] for r in reports),total_outputs=sum(r['clips'] for r in reports),all_inputs_identical=True)
        result['safe_speed_gain']=result['relative_gain']>=.1 and all(c['all_normal'] and c['all_four_captions_identical'] for c in comparisons)
        write(receipt,result)
        return result
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)


def prepare_benchmark(row,old,work,resources):
    from .j_omni_prefetch import prepare_call
    return prepare_call(row,old,work,resources)
