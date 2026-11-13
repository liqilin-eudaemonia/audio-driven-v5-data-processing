from pathlib import Path
import ast,difflib
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];p=ROOT/'src/listen_benchmark/j_omni.py';s=p.read_text();before=s
assert 'preparation_profiles' not in s
s=s.replace('from listen_benchmark.native_caption_quality import generation_failures','from listen_benchmark.native_caption_quality import generation_failures\nfrom listen_benchmark.j_omni_prefetch import PreparationPool')
s=s.replace("work=Path(work);cfg=read(work/'config.json');name=f'{node}-gpu-{gpu}'", "work=Path(work);cfg=read(work/'config.json');name=f'{node}-gpu-{gpu}'\n    baseline=dict(backend='thread',workers=2,prefetch=2,ready_first=False)\n    profile=cfg.get('preparation_profiles',{}).get(name,baseline)\n    if set(profile)!=set(baseline) or type(profile['ready_first']) is not bool or type(profile['prefetch']) is not int or not 1<=profile['prefetch']<=32:raise ValueError('Invalid preparation profile')\n    revision=EXECUTION_REVISION if profile==baseline else EXECUTION_REVISION+'+process-prefetch-v1' ")
s=s.replace('execution_revision=EXECUTION_REVISION)', 'execution_revision=revision,preparation_profile=profile,preparation_wait_s=0.)')
s=s.replace('with ThreadPoolExecutor(max_workers=2) as pool:', "with PreparationPool(profile['backend'],profile['workers']) as pool:")
a="""                rows=sorted(job['rows'],key=lambda r:-r['source']['duration_s']);pending={};next_index=0
                def prep(r):
                    return prepare_input(r,origin['old_work'],work,cfg['nodes'][node]['resource_work'])
                try:
                    for i,row in enumerate(rows):
                        if stop.is_set():break
                        for j in range(next_index,min(len(rows),i+2)):
                            sid=rows[j]['sample_id']
                            if not result_path(work,'caption',sid).exists() and not result_path(work,'held',sid).exists():pending[j]=pool.submit(prep,rows[j])
                            next_index=j+1
                        sid=row['sample_id'];dest=result_path(work,'caption',sid)
                        if dest.exists():
                            old=read(dest)
                            if old['source']!=row['source'] or old['pipeline_id']!=cfg['pipeline_id']:raise ValueError('Existing output provenance mismatch')
                            continue
                        if result_path(work,'held',sid).exists():continue
                        try:prepared=pending.pop(i).result()
"""
b="""                rows=sorted(job['rows'],key=lambda r:-r['source']['duration_s']);todo=[]
                for row in rows:
                    dest=result_path(work,'caption',row['sample_id'])
                    if dest.exists():
                        old=read(dest)
                        if old['source']!=row['source'] or old['pipeline_id']!=cfg['pipeline_id']:raise ValueError('Existing output provenance mismatch')
                    elif not result_path(work,'held',row['sample_id']).exists():todo.append(row)
                stream=pool.stream(todo,origin['old_work'],work,cfg['nodes'][node]['resource_work'],profile['prefetch'],stop,profile['ready_first'])
                try:
                    for row,future,wait_s in stream:
                        if stop.is_set():break
                        sid=row['sample_id'];dest=result_path(work,'caption',sid)
                        try:
                            wait_start=time.monotonic();prepared=future.result();wait_s+=time.monotonic()-wait_start
                            state['preparation_wait_s']+=wait_s
                            if profile['backend']=='process' and prepared['_preparation_metrics']['cuda_visible_devices']!='':raise ValueError('CPU decoder exposed CUDA')
"""
assert a in s;s=s.replace(a,b)
a="""                    # Do not release the source claim while preparation still touches its records.
                    for future in pending.values():
                        try:future.result()
                        except Exception:pass
"""
assert a in s;s=s.replace(a,"                    stream.close()  # drain/cancel preparation before source claim release\n")
s=s.replace('node=node,gpu=gpu,execution_revision=EXECUTION_REVISION,completed_at=',"node=node,gpu=gpu,execution_revision=revision,preparation_profile=profile,preparation_metrics=dict(prepared['_preparation_metrics'],main_wait_s=wait_s),completed_at=")
s=s.replace("if max_clips and state['completed_clips']>=max_clips:stop.set()", "state['stage']='preparing'\n                        if max_clips and state['completed_clips']>=max_clips:stop.set()")
ast.parse(s);(W/'performance_check/j_omni_candidate.py').write_text(s);(W/'performance_check/prefetch_candidate.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='production-v2',tofile='candidate-process-prefetch')))
print('candidate prepared; production unchanged')
