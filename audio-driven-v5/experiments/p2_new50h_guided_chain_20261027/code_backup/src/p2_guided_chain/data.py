"""Read-only strict semantic results -> renderer-compatible derived plan protocol."""
from pathlib import Path
import copy,json,math,sys
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json,observation
from p2_plan_cache.protocol import code_identity,render_clean
from p2_plan_cache import VERSION


def link(path,target):
    path=Path(path);target=Path(target).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        if path.resolve()!=target:raise ValueError('Existing dependency link differs')
    elif path.exists():raise ValueError('Refusing to overwrite dependency')
    else:path.symlink_to(target,target_is_directory=True)


def training_record(result):
    from listen_benchmark.semantic_prompt import VERSION as semantic_version
    if result['schema']!=semantic_version or result['semantic_review']['accepted'] is not True:raise ValueError('Only strict published results may enter training')
    original=result['source_record'];audio=lambda x:sorted((e for e in x['events'] if e['modality']=='audio'),key=lambda e:e['event_id'])
    if audio(result)!=audio(original):raise ValueError('Audio fields changed')
    # Schema compatibility is explicit: source_record is a DERIVED view, never
    # presented as the original integrated annotations.
    return dict(schema='listen-existing-result-fusion-verbatim-v2',sample_id=result['sample_id'],duration_s=result['duration_s'],events=copy.deepcopy(result['events']),
        annotation_view='strict_model_consolidation_published_only',semantic_result_sha256=digest(result),original_record_sha256=result['source_record_sha256'],semantic_protocol_sha256=result['provenance']['protocol_sha256'])


def export_chunk(root,task):
    root=Path(root);spec=read(root/'pipeline.json');folder=Path(task['inputs']).parent;run=Path(task['output'])
    sys.path.insert(0,str(ROOT/'scripts'));from audit_listener_semantic_prompt import audit
    report=audit(Path(task['inputs']),run);atomic_json(folder/'audit.json',report)
    jobs=[json.loads(x) for x in Path(task['inputs']).read_text().splitlines() if x.strip()]
    if (run/'summary.json').exists():
        summary=read(run/'summary.json');rejects={x['job_id']:x for x in summary['rejected']}
        if summary['samples']!=len(jobs) or summary['accepted']!=report['verified']:raise ValueError('Worker publication/summary mismatch')
    else:rejects={}
    data=__import__('p2_listen.config',fromlist=['load']).load(spec['data_config'])
    all_obs={o['sample_id']:o for o in read(Path(data['output'])/'manifests/observations.json')}
    planroot=root/'plans';(planroot/'samples').mkdir(parents=True,exist_ok=True);(planroot/'cache').mkdir(exist_ok=True);(planroot/'locks').mkdir(exist_ok=True)
    work=folder/'plans';work.mkdir(exist_ok=True)
    for name in ('samples','cache','locks'):link(work/name,planroot/name)
    entries={};accepted=[];rejected=[];unresolved=[];derived=[]
    for job in jobs:
        result_path=run/'results'/(job['job_id']+'.json');sid=job['record']['sample_id']
        if result_path.exists():
            result=read(result_path)
            if result['source_record']!=job['record'] or result['source_record_sha256']!=digest(job['record']):raise ValueError('Published source changed')
            record=training_record(result);obs=observation(all_obs[sid],verify=False)
            if abs(obs['requested_duration_s']-record['duration_s'])>1e-6:raise ValueError('Plan duration differs')
            guidance=render_clean(record);item=dict(observation=obs,guidance=guidance)
            atomic_json(planroot/'samples'/(sid+'.json'),item,immutable=True);entries[sid]=digest(item);accepted.append(sid);derived.append(record)
        elif job['job_id'] in rejects:rejected.append(dict(sample_id=sid,**rejects[job['job_id']]))
        else:unresolved.append(job)
    records_path=folder/'derived_records.jsonl';content=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in derived)
    if records_path.exists() and records_path.read_text()!=content:raise ValueError('Derived publication changed')
    records_path.write_text(content)
    sources={str(p):sha(p) for p in [Path(spec['data_config']),Path(data['output'])/'COMPLETE.json',Path(task['inputs']),run/'protocol.json',records_path,Path(__file__)]}
    protocol=dict(schema=VERSION,config=data,sources=sources,code=code_identity(),sample_ids=sorted(accepted),max_text_tokens=8192,max_qwen_tokens=32768,
        plan_source='postprocessed_train_annotation',annotation_text_policy='strict_semantic_published_visual_verbatim_audio',target_annotation_access=True,target_media_access=False,qwen_frozen=True,qwen_lora=False,truncation=False,entries=entries)
    if accepted:atomic_json(work/'protocol.json',protocol,immutable=True)
    result=dict(chunk=task['id'],accepted=accepted,rejected=rejected,unresolved_ids=[j['job_id'] for j in unresolved],work=str(work),audit_sha256=sha(folder/'audit.json'),records=str(records_path),protocol_sha256=digest(protocol) if accepted else None)
    atomic_json(folder/'export.json',result,immutable=True)
    return result,unresolved


def publish_task(root,task):atomic_json(Path(root)/'tasks'/(task['id']+'.json'),task,immutable=True)


def finish_view(root):
    root=Path(root);spec=read(root/'pipeline.json');prepared=read(root/'PREPARED.json');data=__import__('p2_listen.config',fromlist=['load']).load(spec['data_config']);raw=read(Path(data['output'])/'manifests/observations.json')
    outcomes={};entries={};sources={};chunks=[];rejections=[]
    for path in sorted((root/'chunks').glob('*/export.json')):
        export=read(path);chunks.append(str(path))
        p=read(Path(export['work'])/'protocol.json') if export['accepted'] else None
        if p:
            for key,value in p['sources'].items():
                if key in sources and sources[key]!=value:raise ValueError('Conflicting chunk source identity')
                sources[key]=value
        for sid in export['accepted']:
            if sid in outcomes:raise ValueError('Duplicate accepted/rejected member')
            outcomes[sid]='accepted';entries[sid]=p['entries'][sid]
        for item in export['rejected']:
            sid=item['sample_id']
            if sid in outcomes:raise ValueError('Duplicate accepted/rejected member')
            outcomes[sid]='rejected';rejections.append(item)
    for path in (root/'context_rejections').glob('*.json') if (root/'context_rejections').exists() else []:
        item=read(path);sid=item['sample_id']
        if sid in outcomes:raise ValueError('Duplicate context rejection')
        outcomes[sid]='rejected';rejections.append(item)
    if set(outcomes)!={o['sample_id'] for o in raw} or len(outcomes)!=prepared['samples']:raise ValueError('Input coverage is not complete')
    selected=[o for o in raw if outcomes[o['sample_id']]=='accepted']
    if not selected:raise ValueError('No published training members')
    planroot=root/'plans';sample_ids=sorted(entries)
    protocol=dict(schema=VERSION,config=data,sources=sources,code=code_identity(),sample_ids=sample_ids,max_text_tokens=8192,max_qwen_tokens=32768,
        plan_source='postprocessed_train_annotation',annotation_text_policy='strict_semantic_published_visual_verbatim_audio',target_annotation_access=True,target_media_access=False,qwen_frozen=True,qwen_lora=False,truncation=False,entries=entries)
    atomic_json(planroot/'protocol.json',protocol,immutable=True)
    view=root/'view';atomic_json(view/'rejected.json',rejections,immutable=True);atomic_json(view/'observations.json',selected,immutable=True)
    result=dict(samples=len(selected),rejected=len(outcomes)-len(selected),source_samples=len(raw),duration_s=sum(o['requested_duration_s'] for o in selected),steps_per_stage=math.ceil(len(selected)/24),global_batch=24,seed=20261023,observation_sha256=sha(view/'observations.json'),plan_protocol_sha256=digest(protocol),chunks=chunks,policy=spec['policy'])
    atomic_json(view/'COMPLETE.json',result,immutable=True)
    return result
