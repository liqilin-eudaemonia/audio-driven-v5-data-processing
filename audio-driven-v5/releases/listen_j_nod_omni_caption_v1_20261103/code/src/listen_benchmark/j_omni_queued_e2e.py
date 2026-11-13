"""ABBA end-to-end trial of the actual queued production helper."""
import hashlib,time
from pathlib import Path
from .j_annotation import read,write
from .j_omni_prefetch import PreparationPool,prefetch_stream
from .j_omni_queued import groups,ready_groups,generate


def run(llm,processor,cfg,work,node,gpu,folder,stop,progress):
    from .j_omni import fingerprint
    from .j_omni_e2e import prepare_benchmark
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .j_omni_prefetch import cpu_only
    folder=Path(folder);tasks=read(folder/'tasks.json');refs=read(folder/'reference_captions.json')
    profile=cfg['preparation_profiles'][f'{node}-gpu-{gpu}'];assert profile['backend']=='process'
    adaptive=(folder/'adaptive.json').exists() and read(folder/'adaptive.json').get('enabled',False)
    assert adaptive==bool(cfg.get('adaptive_request_groups',{}).get(f'{node}-gpu-{gpu}',False))
    receipt=folder/'complete.json'
    if receipt.exists():
        out=read(receipt)
        if out['pipeline_id']!=cfg['pipeline_id'] or out['tasks_sha256']!=fingerprint(tasks) or out['profile']!=profile or out.get('adaptive',False)!=adaptive:raise ValueError('Queue benchmark configuration changed')
        return out
    run_id=str(time.time_ns());reports=[];lookup={t['row']['sample_id']:t for t in tasks}
    with ProcessPoolExecutor(max_workers=profile['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=cpu_only) as pool:
        for phase,size in enumerate([1,4,4,1]):
            if stop.is_set():return
            llm.reset_prefix_cache();tick=time.monotonic();outputs=[];group_sizes=[]
            def submit(row):return pool.submit(prepare_benchmark,row,lookup[row['sample_id']]['old_work'],str(work),cfg['nodes'][node]['resource_work'])
            stream=prefetch_stream([t['row'] for t in tasks],submit,profile['prefetch'],stop,profile['ready_first'])
            group_iterator=(ready_groups if adaptive and size>1 else groups)(stream,size,stop)
            try:
                for batch in group_iterator:
                    records=[]
                    for row,future,wait_s in batch:
                        p=future.result();t=lookup[row['sample_id']]
                        assert p['prompt']==t['prompt'] and p['_preparation_metrics']['cuda_visible_devices']==''
                        for key,data in [('video_frames_sha256',p['frames']),('audio_encoder_sha256',p['audio'])]:
                            assert hashlib.sha256(data.tobytes()).hexdigest()==t['expected'][key]==p['evidence'][key]
                        records.append(dict(row=row,prepared=p,wait_s=wait_s))
                    group_sizes.append(len(records))
                    gen=generate(llm,processor,records,cfg,folder/run_id/f'phase-{phase}-attempts',paired_ids=size>1)
                    for rec,result in zip(records,gen):
                        a=result['attempts'][-1];sid=rec['row']['sample_id']
                        outputs.append(dict(sample_id=sid,caption=a['caption'],passed=not a['failures'],prompt_tokens_sha256=a['prompt_tokens_sha256'],matches_reference=a['caption']==refs[sid],attempts=result['attempts']))
                        progress(phase,len(outputs),sid)
            finally:
                group_iterator.close();stream.close()
            if stop.is_set():return
            elapsed=time.monotonic()-tick
            report=dict(phase=phase,requests_per_call=size,seconds=elapsed,clips=len(outputs),clips_per_second=len(outputs)/elapsed,request_group_sizes=group_sizes,outputs=outputs)
            write(folder/run_id/f'phase-{phase}.json',report);reports.append(report)
    indexed=[{x['sample_id']:x for x in p['outputs']} for p in reports];comparisons=[]
    for sid in lookup:
        rows=[p[sid] for p in indexed]
        comparisons.append(dict(sample_id=sid,all_four_captions_identical=len({r['caption'] for r in rows})==1,all_prompt_tokens_identical=len({r['prompt_tokens_sha256'] for r in rows})==1,all_normal=all(r['passed'] for r in rows),all_match_reference=all(r['matches_reference'] for r in rows)))
    base=reports[0]['seconds']+reports[3]['seconds'];queued=reports[1]['seconds']+reports[2]['seconds']
    out=dict(at=time.time(),pipeline_id=cfg['pipeline_id'],tasks_sha256=fingerprint(tasks),profile=profile,adaptive=adaptive,run_id=run_id,node=node,gpu=gpu,base_seconds=base,queued_seconds=queued,speedup=base/queued,normal_outputs=sum(x['passed'] for p in reports for x in p['outputs']),total_outputs=sum(len(p['outputs']) for p in reports),comparisons=comparisons,scope='end-to-end preparation/prefetch/request-submission, one GPU, engine capacity1',all_media_and_prompts_identical=True)
    out['safe_speed_gain']=out['speedup']>=1.1 and all(all(r[k] for k in ['all_four_captions_identical','all_prompt_tokens_identical','all_normal','all_match_reference']) for r in comparisons)
    write(receipt,out);return out
