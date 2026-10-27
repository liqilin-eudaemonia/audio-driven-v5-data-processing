"""Durable CPU preparation for the P2 feature follow-on. Never changes raw labels."""
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

from .full_fusion import atomic, read, lock, jsonl, jsonl_writer, file_sha


def build_source_index(run):
    run = Path(run)
    with lock(run/'locks/source-index.lock') as acquired:
        if not acquired or (run/'source_index/COMPLETE.json').exists():
            return
        cfg = read(run/'config.json'); raw = Path(cfg['raw_run'])
        manifests = set()
        for shard in (raw/'queue/shards').glob('*.jsonl'):
            manifests.update(x['source_manifest'] for x in jsonl(shard))
        db = run/'source_index/offsets.sqlite'
        conn = sqlite3.connect(db)
        conn.execute('PRAGMA journal_mode=DELETE')
        conn.execute('CREATE TABLE IF NOT EXISTS offsets (file_id INTEGER,line INTEGER,offset INTEGER, PRIMARY KEY(file_id,line)) WITHOUT ROWID')
        identities = {}
        try:
            for fid, name in enumerate(sorted(manifests)):
                path = Path(name); st = path.stat(); marker = run/f'source_index/file-{fid}.json'
                ident = dict(path=name, size=st.st_size, mtime_ns=st.st_mtime_ns)
                if marker.exists():
                    if read(marker)['identity'] != ident: raise ValueError('Source manifest changed')
                    identities[name] = read(marker); continue
                conn.execute('DELETE FROM offsets WHERE file_id=?',(fid,)); conn.commit()
                batch = []; offset = 0; count = 0; h = hashlib.sha256()
                with path.open('rb') as stream:
                    for count, line in enumerate(stream,1):
                        batch.append((fid,count,offset)); offset += len(line); h.update(line)
                        if len(batch) >= 10000:
                            conn.executemany('INSERT INTO offsets VALUES (?,?,?)',batch);conn.commit();batch=[]
                            atomic(run/'source_index/status.json',dict(heartbeat=time.time(),file=name,lines=count))
                if batch:conn.executemany('INSERT INTO offsets VALUES (?,?,?)',batch);conn.commit()
                if path.stat().st_size != st.st_size or path.stat().st_mtime_ns != st.st_mtime_ns: raise ValueError('Source mutated during indexing')
                info=dict(identity=ident,file_id=fid,lines=count,sha256=h.hexdigest());atomic(marker,info);identities[name]=info
            atomic(run/'source_index/COMPLETE.json',dict(files=identities,utc=time.time(),db_sha256=file_sha(db)))
        finally:conn.close()


def source_row(run, source):
    run=Path(run);index=read(run/'source_index/COMPLETE.json')['files'];info=index[source['source_manifest']]
    path=Path(source['source_manifest']);st=path.stat()
    if st.st_size!=info['identity']['size'] or st.st_mtime_ns!=info['identity']['mtime_ns']:raise ValueError('Manifest changed after indexing')
    with sqlite3.connect(f'file:{run}/source_index/offsets.sqlite?mode=ro',uri=True) as db:
        result=db.execute('SELECT offset FROM offsets WHERE file_id=? AND line=?',(info['file_id'],source['source_line_1based'])).fetchone()
    if result is None:raise ValueError('Source line not indexed')
    with path.open('rb') as f:f.seek(result[0]);line=f.readline()
    if hashlib.sha256(line).hexdigest()!=source['source_line_sha256']:raise ValueError('Source line hash mismatch')
    full=json.loads(line)
    if full['sample_id']!=source['sample_id'] or full['split']!=source['split']:raise ValueError('Source sample/split mismatch')
    if full['sources']['listener']!=source['source']:raise ValueError('Listener video identity differs')
    if full['normalized_audio']['listener']['path']!=source['normalized_audio']['path']:raise ValueError('Listener audio identity differs')
    if full['listener_id']==full['speaker_id']:raise ValueError('Role collision')
    return full


def publish_shard(run, shard):
    run=Path(run);cfg=read(run/'config.json');fusion=Path(cfg['fusion_run']);marker=run/f'state/published/{shard}.json'
    with lock(run/f'locks/publish-{shard}.lock') as acquired:
        if not acquired:return None
        if marker.exists():return read(marker)
        state=read(fusion/f'state/fusion/{shard}.json')
        if not state.get('done'):return None
        for name in [f'labels/{shard}.jsonl',f'evidence/{shard}.jsonl']:
            if file_sha(fusion/name)!=state['files'][name]['sha256']:raise ValueError('Fused source hash mismatch')
        evidence={x['sample_id']:x for x in jsonl(fusion/f'evidence/{shard}.jsonl')}
        fulls={b:{} for b in ['visual','audio']};raw=Path(cfg['raw_run'])
        for branch in fulls:
            for row in jsonl(raw/f'raw/{branch}/results/{shard}.jsonl'):
                if row.get('record_kind')=='result' and row.get('view')=='full':fulls[branch][row['sample_id']]=row
        candidates=list(jsonl(fusion/f'labels/{shard}.jsonl'));jobs=[]
        if len(candidates)!=state['counts']['samples'] or set(evidence)!={c['sample_id'] for c in candidates}:raise ValueError('Shard source IDs differ')
        for offset in range(0,len(candidates),cfg['job_samples']):
            jid=f'{shard}-{offset//cfg["job_samples"]:04d}';records=[]
            for c in candidates[offset:offset+cfg['job_samples']]:
                sid=c['sample_id'];records.append(dict(candidate=c,evidence=evidence[sid],receipts={b:fulls[b].get(sid) for b in fulls}))
            atomic(run/f'jobs/{jid}.json',dict(job_id=jid,shard=shard,fusion_files=state['files'],samples=records));jobs.append(jid)
        value=dict(done=True,shard=shard,jobs=jobs,samples=len(candidates),utc=time.time());atomic(marker,value);return value


def export_media(run, candidate):
    """Reuse the accepted 50-sample crop/frame/audio exporter without debug split overrides."""
    run=Path(run);cfg=read(run/'config.json');sid=candidate['sample_id'];source=candidate['source'];dest=run/'media'/sid
    with lock(run/f'locks/media-sample-{sid}.lock') as acquired:
        if not acquired:raise RuntimeError('Media sample already claimed')
        marker=dest/'COMPLETE.json'
        if marker.exists():
            value=read(marker)
            if value['source_line_sha256']!=source['source_line_sha256']:raise ValueError('Media source identity mismatch')
            for name,h in value['meta']['sha256'].items():
                if file_sha(dest/name)!=h:raise ValueError('Corrupt exported media')
            return value
        # Only incomplete output from this run can be removed; raw media never touched.
        if dest.exists():shutil.rmtree(dest)
        full=source_row(run,source)
        import importlib.util
        path=Path(cfg['project'])/'scripts/export_listen_debug50_media.py'
        spec=importlib.util.spec_from_file_location('p2_media_export',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        obs,target,meta=module.export_one(dict(sample_id=sid,source=source,duration_s=candidate['duration_s']),full,run)
        obs['experiment_split']=obs['official_split']=full['split']
        from p2_listen.dataset import verify_media
        verify_media(obs,target)
        # Native P2 target identity includes this media-validation flag. Keep it
        # stable for later cache reuse; dataset/label formal admission stays separate.
        target['admitted']=True
        value=dict(observation=obs,target=target,meta=meta,source_line_sha256=source['source_line_sha256'],
                   media_verified=True,formal_training_admitted=False)
        atomic(marker,value);return value


def receipt_sample_clock(receipt, candidate, media):
    """Adapt nominal source seconds only after exact source/sample-range checks."""
    if receipt is None:
        return None, None
    source = receipt.get('source', {})
    duration = candidate['duration_s']
    nominal = source.get('duration_s')
    if not isinstance(nominal, (int, float)) or abs(nominal-duration) <= 1e-6:
        return receipt, None
    audio = media['meta']['audio_sources']['listener']
    rate = audio['sample_rate']
    start, end = source.get('source_start_s'), source.get('source_end_s')
    if (source != candidate['source'] or source.get('sample_id') != candidate['sample_id']
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end, nominal))
            or source['normalized_audio']['sample_rate'] != rate
            or abs(nominal-(end-start)) > 1e-6
            or round(start*rate) != audio['start_sample']
            or round(end*rate) != audio['end_sample']
            or abs(duration-(audio['end_sample']-audio['start_sample'])/rate) > 1e-9
            or abs(duration-nominal) > 1/rate+1e-9):
        raise ValueError('Receipt nominal/sample clock identity differs')
    adapted = copy.deepcopy(receipt)
    adapted['source']['duration_s'] = duration
    return adapted, dict(nominal_duration_s=nominal, sample_duration_s=duration,
                         sample_rate=rate, start_sample=audio['start_sample'],
                         end_sample=audio['end_sample'], raw_receipt_modified=False)


def make_label(run, record, media):
    from p2_listen.weak_labels import full_receipt,derive_sample
    from p2_listen.contracts import EVENTS
    run=Path(run);c=read(run/'p2_config.json');candidate=record['candidate'];sid=candidate['sample_id'];d=candidate['duration_s']
    receipts={};problems={};clock_adaptations={}
    for branch in ['visual','audio']:
        r, clock = receipt_sample_clock(record['receipts'][branch],candidate,media)
        if clock:clock_adaptations[branch]=clock
        receipts[branch],problems[branch]=full_receipt([] if r is None else [r],branch,sid,d,media['meta'],media['target'])
    # No artificial full-unknown placeholder; retain explicit uncertainty from evidence.
    raw=dict(sample_id=sid,duration_s=d,events=record['evidence']['original_events'],unknown={k:[] for k in EVENTS})
    label,decision,audit=derive_sample(candidate,raw,c,media['target'],receipts,debug_only=False)
    # This pipeline prepares caches; dataset admission remains an independent decision.
    label['provenance'].update(formal_training_admitted=False,feature_preparation_only=True,
        original_training_admitted=candidate['training_admitted'],raw_completion_status=candidate['raw_completion_status'])
    # Missing semantic evidence stays unknown; it is never converted to background.
    if record['evidence'].get('semantic_issues'):
        for k in EVENTS:label['unknown'][k]=[[0.,d]]
    audit.update(receipt_exclusions=problems,source_duration_clock_adaptations=clock_adaptations,training_started=False)
    atomic(run/f'labels/canonical/{sid}.json',label)
    atomic(run/f'labels/audit/{sid}.json',dict(decision=decision,audit=audit,source_candidate_sha256=hashlib.sha256(json.dumps(candidate,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
    return label


def process_media_job(run,jid,worker,status=None):
    run=Path(run);marker=run/f'state/media/{jid}.json'
    with lock(run/f'locks/media-job-{jid}.lock') as acquired:
        if not acquired:return None
        if marker.exists():return read(marker)
        job=read(run/f'jobs/{jid}.json');rows=[];failures=[]
        for record in job['samples']:
            sid=record['candidate']['sample_id']
            if status:status('media',job_id=jid,sample_id=sid)
            try:
                media=export_media(run,record['candidate']);label=make_label(run,record,media)
                rows.append(dict(sample_id=sid,observation=media['observation'],target=media['target'],label_training_admitted=label['training_admitted']))
                alert=run/f'alerts/sample-{sid}.json'
                if alert.exists():atomic(alert,dict(read(alert),resolved_at=time.time()))
            except Exception as exc:
                failure=dict(sample_id=sid,stage='media',error=repr(exc),worker=worker,utc=time.time());atomic(run/f'alerts/sample-{sid}.json',failure);raise
        path=run/f'manifests/shards/{jid}.json';atomic(path,rows)
        value=dict(done=True,job_id=jid,samples=len(rows),manifest_sha256=file_sha(path),worker=worker,utc=time.time());atomic(marker,value);return value


def select_jobs(run,stage,rank=0):
    run=Path(run);source=run/'jobs' if stage=='media' else run/'state/media'
    queue=read(run/'queues.json')
    values=queue.get(stage)
    if values is None:values=sorted(p.stem for p in source.glob('*.json') if not (run/f'state/{stage}/{p.stem}.json').exists())
    if values:offset=rank%len(values);values=values[offset:]+values[:offset]
    return values
