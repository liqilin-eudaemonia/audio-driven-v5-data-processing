"""Durable shard adapter for raw annotations -> semantic categories -> verbatim fusion.

No media inference or training admission. Original raw files are read-only.
"""
import contextlib,copy,fcntl,hashlib,json,math,os,time
from collections import Counter
from pathlib import Path
from .semantic_mapping import VERSION as MAPPING_VERSION, request_prompt,validate_answer,enrich_sample
from .response_review import alignment_track
from .existing_result_fusion import fuse,VERSION as FUSION_VERSION

VERSION='listen-full-verbatim-fusion-v1'
BRANCHES=('visual','audio','asr')

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:yield False;return
        try:yield True
        finally:fcntl.flock(f,fcntl.LOCK_UN)

@contextlib.contextmanager
def jsonl_writer(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temp.open('w') as f:
            yield lambda x:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()

def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def views(duration):
    out=[('full',0.,duration)]
    for i in range(math.ceil(max(0.,duration-3.))+1):
        a,b=float(i),min(duration,i+3.)
        if (a,b)!=(0.,duration):out.append((f'w{i:03d}',a,b))
    return out

def ready_raw(raw,shard):
    return all(read(Path(raw)/f'raw/{b}/state/{shard}.json').get('done') for b in BRANCHES)

def semantic_job(modality,text,model_signature):
    payload=dict(modality=modality,primary_description=text,original_sources=[dict(source_id='raw',description=text,source='qwen3_vl' if modality=='visual' else 'omni_audio')])
    prompt=request_prompt({'item':payload})
    key=digest(dict(payload=payload,prompt=prompt,mapping_version=MAPPING_VERSION,model_signature=model_signature))
    return dict(key=key,payload=payload,prompt_sha256=digest(prompt))

def cache_path(run,key,failed=False):return Path(run)/('semantic_failures' if failed else 'semantic_cache')/key[:2]/(key+'.json')

def cached(run,job,allow_failure=True):
    for failed in ([False,True] if allow_failure else [False]):
        path=cache_path(run,job['key'],failed)
        if not path.exists():continue
        r=read(path)
        if r.get('key')!=job['key'] or r.get('payload')!=job['payload'] or r.get('prompt_sha256')!=job['prompt_sha256']:
            raise ValueError('Cache identity mismatch: '+str(path))
        if failed:
            if r.get('valid') is not False:raise ValueError('Invalid failure cache')
        else:
            if r.get('valid') is not True or r.get('finish_reason')!='stop':raise ValueError('Invalid successful cache')
            validate_answer(job['payload'],r['answer'])
        return r
    return None

def cache_put(run,record):
    path=cache_path(run,record['key'],not record['valid']);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as f:
        json.dump(record,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,path) # First complete result wins; never overwrite another worker.
    except FileExistsError:pass
    finally:temp.unlink()

def prepare_shard(run,shard):
    run=Path(run);cfg=read(run/'config.json');raw=Path(cfg['raw_run']);marker=run/f'state/ingest/{shard}.json'
    if marker.exists():return read(marker)
    if not ready_raw(raw,shard):return None
    with lock(run/f'locks/ingest/{shard}.lock') as acquired:
        if not acquired:return None
        if marker.exists():return read(marker)
        rows=list(jsonl(raw/f'queue/shards/{shard}.jsonl'));ids={r['sample_id'] for r in rows}
        if len(rows)!=len(ids):raise ValueError('Duplicate sample in raw queue')
        latest={b:{} for b in BRANCHES};rawmeta={};rawissues=[]
        for b in BRANCHES:
            p=raw/f'raw/{b}/results/{shard}.jsonl';before=p.stat();h=hashlib.sha256();lines=0
            with p.open('rb') as f:
                for lines,line in enumerate(f,1):
                    h.update(line)
                    try:r=json.loads(line)
                    except Exception as e:raise ValueError(f'Malformed immutable raw line {p}:{lines}: {e}')
                    if r.get('record_kind')!='result':continue
                    if r.get('sample_id') not in ids:
                        rawissues.append(dict(branch=b,reason='raw_sample_not_in_queue',line=lines));continue
                    r['_raw_reference']=dict(file=str(p),line_1based=lines,line_sha256=hashlib.sha256(line).hexdigest())
                    latest[b][r['request_id']]=r
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Raw changed during ingest')
            rawmeta[b]=dict(path=str(p),sha256=h.hexdigest(),bytes=after.st_size,lines=lines)
        repair_file=raw/'raw/asr_alignment_repairs/results.jsonl'
        repairs={r['request_id']:r for r in jsonl(repair_file)} if repair_file.exists() else {}
        jobs={};stats=Counter();actual_expected=0;queue_expected=0
        with jsonl_writer(run/f'input/{shard}.jsonl') as emit:
            for row in rows:
                sid=row['sample_id'];rate=row['normalized_audio']['sample_rate']
                exact_duration=(round(row['source_end_s']*rate)-round(row['source_start_s']*rate))/rate
                duration=round(exact_duration,9)
                expected={sid+'@'+v for v,_,_ in views(exact_duration)};actual_expected+=len(expected);queue_expected+=len(views(row['duration_s']))
                events=[];sources=[];origins={};tracks=[];refs=[];issues=[];event_jobs={}
                for b in BRANCHES:
                    present={rid for rid in latest[b] if rid.startswith(sid+'@')};missing=expected-present
                    for rid in sorted(missing):issues.append(dict(branch=b,request_id=rid,reason='missing_expected_raw_result'))
                    for rid in sorted(present):
                        r=latest[b][rid];refs.append(r['_raw_reference'])
                        if r['source']['source_line_sha256']!=row['source_line_sha256']:
                            issues.append(dict(branch=b,request_id=rid,reason='source_identity_mismatch'));continue
                        if rid not in expected:
                            issues.append(dict(branch=b,request_id=rid,reason='unexpected_raw_view',status=r.get('status')));continue
                        status=r.get('status');repair=repairs.get(rid) if b=='asr' else None
                        response=r.get('raw_response',{})
                        if status=='alignment_failed' and repair and repair.get('status')=='complete':
                            if response.get('text') and repair['raw_response']['text']!=response['text']:raise ValueError('Repair transcript changed')
                            response=repair['raw_response'];status='complete';refs.append(dict(file=str(repair_file),request_id=rid,repair=True));stats['asr_repairs_used']+=1
                        if status!=('complete' if b=='asr' else 'schema_pass'):
                            issues.append(dict(branch=b,request_id=rid,reason='raw_request_failed',status=status,error=r.get('error')));continue
                        if abs(r.get('parent_duration_s',duration)-duration)>1/rate+1e-8:
                            issues.append(dict(branch=b,request_id=rid,reason='duration_mismatch'));continue
                        if b=='asr':
                            record=dict(response,sample_id=sid if r['view']=='full' else sid+'__'+r['view'],status=status,input_evidence={} if r['view']=='full' else dict(window=dict(start_s=r['window_start_s'],end_s=r['window_end_s'])))
                            tracks.append(alignment_track(record,duration));continue
                        for i,event in enumerate(r['parsed']['events']):
                            start=round(r['window_start_s']+event['start'],9);end=round(r['window_start_s']+event['end'],9)
                            if event['modality']!=b or not 0<=start<end<=duration or not isinstance(event['behavior'],str):
                                issues.append(dict(branch=b,request_id=rid,event_index=i,reason='invalid_global_event_interval_or_modality'));continue
                            eid=f'{b}-{r["view"]}-{i:04d}';cid='source-'+eid
                            events.append(dict(event_id=eid,start=start,end=end,modality=b,behavior=event['behavior'],review_flags=['raw_model_unverified']))
                            source=dict(start=start,end=end,modality=b,behavior=event['behavior'],source='qwen3_vl' if b=='visual' else 'omni_audio',window_id=rid,window=dict(start_s=r['window_start_s'],end_s=r['window_end_s']),request_id=rid)
                            sources.append(dict(id=cid,start=start,end=end,modality=b,behavior=event['behavior'],source=source['source'],sources=[source],valid=True,human_verified=False))
                            origins[eid]=dict(source_candidate_ids=[cid],boundary_method='raw_observed_interval_no_union')
                            job=semantic_job(b,event['behavior'],cfg['model_signature']);jobs[job['key']]=job;event_jobs[eid]=job['key']
                stats['samples']+=1;stats['observations']+=len(events);stats['samples_with_raw_issues']+=bool(issues);stats['raw_issues']+=len(issues)
                emit(dict(sample_id=sid,label=dict(sample_id=sid,duration_s=duration,time_origin='clip_start_seconds',events=events),provenance=dict(source_candidates=sources,events=origins,excluded=[]),tracks=sorted(tracks,key=lambda t:t['pass_id']),event_jobs=event_jobs,source=row,raw_references=refs,raw_issues=issues))
        with jsonl_writer(run/f'jobs/{shard}.jsonl') as emit:
            for key in sorted(jobs):emit(jobs[key])
        result=dict(done=True,shard=shard,utc=time.time(),counts=dict(stats),unique_semantic_jobs=len(jobs),raw_files=rawmeta,queue_sha256=file_sha(raw/f'queue/shards/{shard}.jsonl'),actual_expected_views_per_branch=actual_expected,index_expected_views_per_branch=queue_expected,raw_unassigned_issues=rawissues,input_sha256=file_sha(run/f'input/{shard}.jsonl'),jobs_sha256=file_sha(run/f'jobs/{shard}.jsonl'))
        atomic(marker,result);return result

def semantic_state(run,shard,worker):
    run=Path(run);done=failed=0
    for job in jsonl(run/f'jobs/{shard}.jsonl'):
        record=cached(run,job)
        if record is None:return None
        done+=bool(record['valid']);failed+=not record['valid']
    r=dict(done=True,shard=shard,worker=worker,utc=time.time(),valid=done,failed=failed)
    atomic(run/f'state/semantic/{shard}.json',r);return r

def fuse_shard(run,shard):
    run=Path(run);marker=run/f'state/fusion/{shard}.json'
    if marker.exists():return read(marker)
    if not read(run/f'state/semantic/{shard}.json').get('done'):return None
    with lock(run/f'locks/fusion/{shard}.lock') as acquired:
        if not acquired:return None
        if marker.exists():return read(marker)
        cfg=read(run/'config.json');jobs={j['key']:j for j in jsonl(run/f'jobs/{shard}.jsonl')};cache={k:cached(run,j) for k,j in jobs.items()};stats=Counter();classes=Counter()
        if any(r is None for r in cache.values()):raise ValueError('Semantic completion without all records')
        with jsonl_writer(run/f'labels/{shard}.jsonl') as output,jsonl_writer(run/f'evidence/{shard}.jsonl') as evidence,jsonl_writer(run/f'issues/{shard}.jsonl') as issues_out:
            for row in jsonl(run/f'input/{shard}.jsonl'):
                label=row['label'];answers={};semantic_issues=[]
                for event in label['events']:
                    eid=event['event_id'];record=cache[row['event_jobs'][eid]];cid='source-'+eid
                    payload=dict(modality=event['modality'],primary_description=event['behavior'],original_sources=[dict(source_id=cid,description=event['behavior'],source='qwen3_vl' if event['modality']=='visual' else 'omni_audio')])
                    answer=copy.deepcopy(record['answer']) if record['valid'] else dict(categories=[],uncertain=True,reason='Semantic inference failed; no category inferred.',evidence_source_ids=['primary'])
                    answer['evidence_source_ids']=[cid if x=='raw' else x for x in answer['evidence_source_ids']]
                    if not record['valid']:semantic_issues.append(dict(event_id=eid,key=record['key'],reason='semantic_inference_failed'))
                    answers[eid]=dict(payload=payload,answer=answer,response_file=str(cache_path(run,record['key'],not record['valid'])),item_id=record['key'])
                enriched=enrich_sample(label,row['provenance'],answers);result=fuse(enriched,enriched['decisions'],row['tracks'])
                originals=enriched['events'];ids={e['event_id'] for e in originals};covered={x['observation_id'] for x in result['evidence_links']+result['unresolved_observations']};covered.update(i for e in result['events'] for i in e['source_observation_ids'])
                if ids!=covered:raise AssertionError('Observation lost during fusion')
                for a,b in zip(label['events'],originals):
                    if any(a[k]!=b[k] for k in ['behavior','start','end','modality']):raise AssertionError('Raw content rewritten')
                for e in result['events']:
                    if not 0<=e['start']<e['end']<=label['duration_s']:raise AssertionError('Output boundary invalid')
                    if 'timing_evidence' in e:
                        if e['description']!=e['text'] or not any((m['start'],m['end'])==(e['start'],e['end']) for m in e['timing_evidence']['members']):raise AssertionError('Invented speech interval/text')
                    elif e['description'] not in e['original_descriptions']:raise AssertionError('Invented description')
                    classes.update(e['categories'])
                raw_issues=row['raw_issues'];failures=result['automatic_fusion']['alignment_failures']
                status='partial_raw' if raw_issues else 'semantic_issues' if semantic_issues else 'complete_candidates'
                result.update(pipeline_version=VERSION,raw_completion_status=status,source=row['source'],evidence_file=f'evidence/{shard}.jsonl',raw_issue_count=len(raw_issues),semantic_issue_count=len(semantic_issues),mapping_version=MAPPING_VERSION,content_policy='verbatim_sources_no_semantic_replacement')
                evidence(dict(sample_id=row['sample_id'],original_events=originals,decisions=enriched['decisions'],alignment_tracks=row['tracks'],raw_references=row['raw_references'],raw_issues=raw_issues,semantic_issues=semantic_issues))
                for k in ['pre_correction_events','alignment_tracks','token_clusters']:result.pop(k,None)
                output(result)
                if raw_issues or semantic_issues or failures:issues_out(dict(sample_id=row['sample_id'],raw=raw_issues,semantic=semantic_issues,unlocalized_asr=failures))
                stats['samples']+=1;stats['events']+=len(result['events']);stats['original_observations']+=len(originals);stats['without_candidates']+=not bool(result['events']);stats['samples_with_raw_issues']+=bool(raw_issues);stats['semantic_failed_events']+=len(semantic_issues);stats['unlocalized_asr_units']+=len(failures)
        if stats['samples']!=read(run/f'state/ingest/{shard}.json')['counts']['samples']:raise AssertionError('Sample count changed')
        result=dict(done=True,shard=shard,utc=time.time(),counts=dict(stats),per_class=dict(classes),files={str(p.relative_to(run)):dict(sha256=file_sha(p),bytes=p.stat().st_size) for p in [run/f'labels/{shard}.jsonl',run/f'evidence/{shard}.jsonl',run/f'issues/{shard}.jsonl']})
        atomic(marker,result);return result
