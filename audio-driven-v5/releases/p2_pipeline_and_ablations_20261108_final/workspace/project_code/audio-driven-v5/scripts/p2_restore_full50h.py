#!/usr/bin/env python3
"""Recover every new50h member; strict safe components, original-event fallback."""
from pathlib import Path
import sys,json,copy,math,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json,observation
from p2_plan_cache.protocol import render_clean,code_identity
from p2_plan_cache import VERSION
from p2_guided_chain.data import link,publish_task
from p2_full50h_restore.annotations import recover


def recover_job(job,runs,protocols):
    record=job["record"];jid=job["job_id"]
    candidates=[];reviews=[]
    for run in reversed(runs):
        proto=protocols[str(run)];protocol_sha=digest(proto)
        for attempt in (2,1,0):
            rp=run/'attempts'/f'{jid}.review.{attempt}.json'
            if rp.exists():reviews.append(dict(path=str(rp),sha256=sha(rp)))
            path=run/'attempts'/f'{jid}.proposal.{attempt}.json'
            if not path.exists():continue
            value=read(path)
            if value.get('protocol_sha256')!=protocol_sha or value.get('job_id')!=jid or value.get('stage')!='proposal' or value.get('attempt')!=attempt:raise ValueError('Attempt provenance mismatch')
            if value.get('finish_reason')!='stop':continue
            try:plan=json.loads(value['raw'])
            except (ValueError,TypeError):continue
            candidates.append((path,plan))
    # Prefer the candidate preserving the most safely checked sources; ties
    # retain latest attempt/order. Every unresolved group remains verbatim.
    best_events,best=recover(record,None);selected=None
    for path,plan in candidates:
        events,audit=recover(record,plan)
        if len(audit['safe_source_ids'])>len(best['safe_source_ids']):best_events,best,selected=events,audit,path
        if not best['fallback_source_ids']:break
    return job,best_events,best,selected,reviews


def recovered_jobs(path,index,protocols):
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    with ProcessPoolExecutor(max_workers=16) as workers:
        pending=deque()
        for line in Path(path).open():
            if not line.strip():continue
            job=json.loads(line);runs=index.get(job["job_id"],[])
            pending.append(workers.submit(recover_job,job,runs,{str(r):protocols[str(r)] for r in runs}))
            if len(pending)>=48:yield pending.popleft().result()
        while pending:yield pending.popleft().result()


def build(old,root):
    spec=read(root/'pipeline.json');raw=__import__('p2_listen.config',fromlist=['load']).load(spec['data_config']);obs=read(Path(raw['output'])/'manifests/observations.json');byid={o['sample_id']:o for o in obs}
    index={};protocols={}
    for path in sorted((old/'tasks').glob('*.json')):
        t=read(path)
        if t['kind']!='semantic':continue
        run=Path(t['output'])
        if not (run/'protocol.json').exists():continue
        protocols[str(run)]=read(run/'protocol.json')
        jobs=[json.loads(line) for line in Path(t['inputs']).open() if line.strip()]
        if digest(jobs)!=protocols[str(run)]['input_sha256']:raise ValueError('Worker input identity changed')
        for job in jobs:index.setdefault(job['job_id'],[]).append(run)
    entries={};sources={};stats=dict(total=0,reused_previous=0,partial_or_original=0,hard_only=0,visual_sources=0,safe_visual_sources=0,original_visual_sources=0);pending=[];chunk=0
    base_sources={str(Path(spec['data_config'])):sha(spec['data_config']),str(Path(raw['output'])/'COMPLETE.json'):sha(Path(raw['output'])/'COMPLETE.json'),str(ROOT/'src/p2_full50h_restore/annotations.py'):sha(ROOT/'src/p2_full50h_restore/annotations.py'),str(Path(__file__)):sha(__file__)}
    def protocol(ids):return dict(schema=VERSION,config=raw,sources=base_sources,code=code_identity(),sample_ids=sorted(ids),entries={sid:entries[sid] for sid in sorted(ids)},max_text_tokens=8192,max_qwen_tokens=32768,plan_source='postprocessed_train_annotation',annotation_text_policy='safe_hard_checked_components_else_original; model_review_record_only',target_annotation_access=True,target_media_access=False,qwen_frozen=True,qwen_lora=False,truncation=False)
    def flush():
        nonlocal chunk,pending
        if not pending:return
        ids=pending;pending=[];work=root/'cache_chunks'/f'{chunk:04d}';work.mkdir(parents=True,exist_ok=True)
        for name in ('samples','cache','locks'):link(work/name,root/'plans'/name)
        atomic_json(work/'protocol.json',protocol(ids),immutable=True)
        for part in ('text','input_plan'):publish_task(root,dict(id=f'cache_{chunk:04d}_{part}',kind=part,work=str(work),priority=10))
        chunk+=1
    for job,best_events,best,selected,reviews in recovered_jobs(spec['inputs'],index,protocols):
        record=job['record'];sid=record['sample_id'];jid=job['job_id']
        audio=lambda events:sorted((e for e in events if e['modality']=='audio'),key=lambda e:e['event_id'])
        if audio(best_events)!=audio(record['events']):raise ValueError('Audio changed')
        old_item_path=old/'plans/samples'/(sid+'.json');reuse=False
        if old_item_path.exists():
            previous=read(old_item_path)
            if previous['guidance']['source_record']['events']==best_events:reuse=True;item=previous
        if not reuse:
            derived=copy.deepcopy(record);derived.update(events=best_events,annotation_view='full50h_hard_check_with_original_fallback',original_record_sha256=digest(record),recovery_policy='safe_components_else_verbatim_original; semantic_review_record_only')
            item=dict(observation=observation(byid[sid],verify=False),guidance=render_clean(derived))
        atomic_json(root/'plans/samples'/(sid+'.json'),item,immutable=True);entries[sid]=digest(item)
        audit=dict(best,sample_id=sid,original_record_sha256=digest(record),selected_proposal=str(selected) if selected else None,selected_proposal_sha256=sha(selected) if selected else None,review_policy='record_only',review_records=reviews,reused_previous_condition=reuse,events_sha256=digest(best_events),audio_unchanged=True)
        atomic_json(root/'recovery'/ (sid+'.json'),audit,immutable=True)
        if reuse:
            for part in ('text','input_plan'):link(root/'plans/cache'/part/sid,old/'plans/cache'/part/sid)
            stats['reused_previous']+=1
        else:pending.append(sid)
        stats['total']+=1;stats['visual_sources']+=len(best['safe_source_ids'])+len(best['fallback_source_ids']);stats['safe_visual_sources']+=len(best['safe_source_ids']);stats['original_visual_sources']+=len(best['fallback_source_ids']);stats['partial_or_original' if best['fallback_source_ids'] else 'hard_only']+=1
        if len(pending)>=128:flush()
        if stats['total']%100==0:atomic_json(root/'RECOVERY_PROGRESS.json',dict(**stats,cache_chunks=chunk));print(json.dumps(stats),flush=True)
    flush()
    if set(entries)!=set(byid) or len(entries)!=29975:raise ValueError('Full29975 membership not preserved')
    atomic_json(root/'plans/protocol.json',protocol(list(entries)),immutable=True)
    atomic_json(root/'view/observations.json',obs,immutable=True);atomic_json(root/'view/rejected.json',[],immutable=True)
    duration=sum(o['requested_duration_s'] for o in obs)
    if abs(duration-180000.532)>1.:raise ValueError('Original50h duration changed')
    manifest=dict(samples=len(obs),rejected=0,source_samples=len(obs),duration_s=duration,steps_per_stage=math.ceil(len(obs)/24),global_batch=24,seed=20261023,observation_sha256=sha(root/'view/observations.json'),plan_protocol_sha256=digest(protocol(list(entries))),policy=spec['policy'])
    atomic_json(root/'view/COMPLETE.json',manifest,immutable=True);atomic_json(root/'RECOVERY_COMPLETE.json',dict(**stats,cache_chunks=chunk,seconds=duration),immutable=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--old',required=True,type=Path);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    import fcntl
    with (a.root/'recovery_build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (a.root/'RECOVERY_COMPLETE.json').exists():build(a.old.resolve(),a.root.resolve())
