"""ABBA, same GPU and samples: original source-by-source vs optimized preparation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import chain
import os,json,multiprocessing,time,hashlib
from .j_annotation import read,write
from .j_omni_prefetch import cpu_only,prepare_call,prefetch_stream
from .j_omni_queued import ready_groups,generate


def initialize(optim):
    cpu_only();os.environ['J_OMNI_BOUNDED_CROP_V1']='1'
    os.environ['J_OMNI_INPUT_OPTIMIZATIONS']=json.dumps(optim)


def warm():
    import numpy,scipy,av
    return os.getpid()


def run(llm,processor,cfg,work,node,gpu,folder,stop,progress):
    from .j_omni import fingerprint
    folder=Path(folder);tasks=read(folder/'tasks.json');report_path=folder/'complete.json'
    if report_path.exists():
        result=read(report_path)
        if result['tasks_sha256']!=fingerprint(tasks):raise ValueError('Input benchmark task drift')
        return result
    optim=dict(fast_crop=True,shared_decode=True,cross_source_prefetch=True,max_sources=4)
    profiles=[{},optim];pools=[];reports=[];resources=cfg['nodes'][node]['resource_work'];run_id=str(time.time_ns())
    try:
        for profile in profiles:
            pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(profile,));pools.append(pool)
            for f in [pool.submit(warm) for _ in range(4)]:f.result()
        for phase,arm in enumerate([0,1,1,0]):
            if stop.is_set():return None
            root=folder/'runs'/run_id/f'phase-{phase}';root.mkdir(parents=True,exist_ok=True)
            c=json.loads(json.dumps(cfg));c['bounded_crop_input']['nod_cache']=str(root/'nod_cache');write(root/'config.json',c)
            (root/'official_metadata').symlink_to(Path(work)/'official_metadata',target_is_directory=True)
            lookup={t['row']['sample_id']:t for t in tasks};groups={}
            for t in tasks:groups.setdefault(t['row']['source']['pair'],[]).append(t['row'])
            def submit(row):return pools[arm].submit(prepare_call,row,lookup[row['sample_id']]['old_work'],root,resources)
            group_values=list(groups.values())
            streams=([prefetch_stream(list(chain.from_iterable(group_values[i:i+4])),submit,8,stop,True) for i in range(0,len(group_values),4)] if arm else
                     [prefetch_stream(rows,submit,8,stop,True) for rows in groups.values()])
            started=time.monotonic();outputs=[]
            try:
                for stream in streams:
                    iterator=ready_groups(stream,4,stop)
                    try:
                        for batch in iterator:
                            records=[]
                            for row,future,wait_s in batch:
                                p=future.result();expected=lookup[row['sample_id']]['expected']
                                actual=dict(prompt=p['prompt'],video=hashlib.sha256(p['frames']).hexdigest(),audio=hashlib.sha256(p['audio']).hexdigest())
                                if actual!=expected:raise ValueError('Benchmark model input differs: '+row['sample_id'])
                                records.append(dict(row=row,prepared=p,wait_s=wait_s))
                            generated=generate(llm,processor,records,cfg,root/'attempts')
                            for rec,gen in zip(records,generated):
                                att=gen['attempts'][-1]
                                outputs.append(dict(sample_id=rec['row']['sample_id'],caption=att['caption'],passed=not att['failures'],prompt_tokens_sha256=att['prompt_tokens_sha256'],preparation=rec['prepared']['_preparation_metrics']))
                            progress(phase,len(outputs),records[-1]['row']['sample_id'])
                    finally:iterator.close();stream.close()
            finally:
                for stream in streams:stream.close()
            report=dict(phase=phase,arm=arm,seconds=time.monotonic()-started,outputs=outputs)
            if len(outputs)!=len(tasks):raise ValueError('Incomplete benchmark phase')
            write(root/'report.json',report);reports.append(report)
        comparisons=[]
        for sid in lookup:
            entries=[next(x for x in r['outputs'] if x['sample_id']==sid) for r in reports]
            comparisons.append(dict(sample_id=sid,all_passed=all(x['passed'] for x in entries),same_caption=len({x['caption'] for x in entries})==1,same_prompt_tokens=len({x['prompt_tokens_sha256'] for x in entries})==1))
        before=reports[0]['seconds']+reports[3]['seconds'];after=reports[1]['seconds']+reports[2]['seconds']
        result=dict(at=time.time(),run_id=run_id,tasks_sha256=fingerprint(tasks),clips_per_phase=len(tasks),phases=4,
            original_seconds=before,optimized_seconds=after,speedup=before/after,all_inputs_identical=True,comparisons=comparisons,
            safe_speed_gain=before/after>1.05 and all(x['all_passed'] and x['same_caption'] and x['same_prompt_tokens'] for x in comparisons))
        write(report_path,result);return result
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)
