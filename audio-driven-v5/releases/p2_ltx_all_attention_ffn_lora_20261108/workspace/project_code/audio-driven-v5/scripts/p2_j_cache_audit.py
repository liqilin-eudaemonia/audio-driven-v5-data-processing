#!/usr/bin/env python3
"""Monitor extraction and audit completed receipts without re-encoding data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,digest,read


def status(work):
    work=Path(work);now=time.time()
    rows=[read(p) for p in (work/'workers').glob('*.json')]
    phases={};config=read(work/'cluster_config.json')
    for phase in ('text','features'):
        group=[r for r in rows if r['phase']==phase]
        jobs=read(work/'jobs'/(phase+'.json'))
        done={p.stem for p in (work/'state'/phase).glob('*.json')}
        if not done.issubset(set(jobs)):raise ValueError('Unexpected task receipt')
        chunk=config['text_chunk' if phase=='text' else 'feature_chunk']
        count=len(done)*chunk
        if jobs[-1] in done:count+=len(read(work/'jobs'/phase/(jobs[-1]+'.json')))-chunk
        phases[phase]=dict(samples=count,finished_jobs=len(done),jobs=len(jobs),
            states=dict(Counter(r['status'] for r in group)),
            quarantined_jobs=[read(p) for p in sorted((work/'quarantine'/phase).glob('*.json'))],
            failures=[dict(worker=r['worker'],error=r['error']) for r in group if r['status']=='failed'])
    resources={}
    for path in sorted((work/'telemetry').glob('*.jsonl')):
        snapshots=[json.loads(line) for line in path.read_text().splitlines()]
        recent=[s for s in snapshots if s['time']>=now-60]
        values=[v[2] for s in recent for v in s['gpus']]
        resources[path.stem]=dict(mean_gpu_60s=sum(values)/len(values) if values else None,
            last_sample_age_s=now-snapshots[-1]['time'],phase=snapshots[-1]['phase'],
            latest_utilization=[v[2] for v in snapshots[-1]['gpus']])
    value=dict(time=now,phases=phases,resources=resources)
    atomic_json(work/'status.json',value)
    with (work/'status_history.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
    return value


_reader=None


def audit_sample(sid):
    from p2_j_cache.data import sample
    from p2_j_cache.runtime import cache_identity
    reader=_reader;work=reader.work;m=reader.manifest;item=sample(work,m,sid);total=0
    if item['split']!='train':raise ValueError('Unexpected dev/test in 500h cache')
    for part in ('input','input_plan','text_guided','vae'):
        path=work/'cache'/part/sid
        marker=read(path/'complete.json')
        expected=cache_identity(m,item,part,reader.assets[part])
        if marker['identity']!=digest(expected) or marker['contract']!=expected:
            raise ValueError('Cache identity differs: '+str(path))
        if len(marker['payload_sha256'])!=64:raise ValueError('Missing writer checksum')
        size=(path/'payload.pt').stat().st_size
        if size==0:raise ValueError('Empty cache payload')
        total+=size
    audit=read(work/'media_audit'/(sid+'.json'))
    if audit.get('video_sampling_override')!=m.get('video_sampling_overrides',{}).get(sid):
        raise ValueError('Reviewed video timing policy differs: '+sid)
    if (audit['sample_id']!=sid or audit['split']!='train' or
        audit['final_crop']!=item['source']['video_transform']['crop'] or
        audit['caption_sha256']!=item['caption_sha256'] or
        not audit['reference_from_first_frame'] or not audit['source_video_used']):
        raise ValueError('Source audit differs: '+sid)
    return dict(bytes=total,nonzero=not audit['listener_full_silence'],
        training_admitted=item['training_admitted'],historical_crop=item['caption_crop']!=audit['final_crop'])


def audit(work,processes):
    global _reader
    import torch
    from p2_j_cache.runtime import CacheReader
    torch.set_num_threads(1);started=time.time();work=Path(work);_reader=CacheReader(work)
    m=_reader.manifest;ids=m['sample_ids'];expected=set(ids)
    for phase in ('text','features'):
        jobs=read(work/'jobs'/(phase+'.json'));covered=[]
        for jid in jobs:
            receipt=read(work/'state'/phase/(jid+'.json'))
            assigned=read(work/'jobs'/phase/(jid+'.json'))
            if receipt['sample_ids']!=assigned or not receipt['success'] or receipt['samples']!=len(assigned):
                raise ValueError('Invalid task receipt')
            covered.extend(assigned)
        if len(covered)!=len(ids) or set(covered)!=expected:raise ValueError('Incomplete or duplicate coverage: '+phase)
    # Each writer has already hashed and reloaded its payload and compared every
    # tensor. This independent pass checks complete coverage and contract bindings.
    _reader.part(None,'text')
    sums=Counter();count=0
    with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork')) as pool:
        for start in range(0,len(ids),8192):
            for result in pool.map(audit_sample,ids[start:start+8192],chunksize=32):
                sums.update(result);count+=1
            atomic_json(work/'audit_progress.json',dict(samples=count,total=len(ids),elapsed_s=time.time()-started))
    result=dict(success=True,samples=count,hours=m['selection_summary']['hours'],
        parts={p:count for p in ('input','input_plan','text_guided','vae')},payload_bytes=sums['bytes'],
        nonzero_listener_audio=sums['nonzero'],silent_listener_audio=count-sums['nonzero'],
        upstream_training_admitted=sums['training_admitted'],historical_caption_crop_differs=sums['historical_crop'],
        original_caption_and_admission_preserved=True,event_intervals_generated=False,
        reviewed_video_sampling_overrides=list(m.get('video_sampling_overrides',{})),
        writer_validation='SHA256 plus full tensor roundtrip equality for every payload',
        final_validation='Independent complete receipt coverage and all cache/source identity bindings',
        elapsed_s=time.time()-started,finished_at=time.time())
    atomic_json(work/'COMPLETE.json',result,immutable=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['status','audit','watch'])
    parser.add_argument('--work',required=True);parser.add_argument('--processes',type=int,default=64);a=parser.parse_args()
    if a.command=='watch':
        work=Path(a.work)
        while True:
            if (work/'COMPLETE.json').exists():result=read(work/'COMPLETE.json');break
            value=status(work);phases=value['phases']
            print(json.dumps(dict(time=value['time'],text=phases['text']['samples'],features=phases['features']['samples'],
                gpu_mean={n:r['mean_gpu_60s'] for n,r in value['resources'].items()})),flush=True)
            failures=[r for phase in phases.values() for r in phase['failures']]
            if failures:print(json.dumps(dict(worker_failures=failures)),flush=True)
            quarantined=[r for phase in phases.values() for r in phase['quarantined_jobs']]
            if quarantined:print(json.dumps(dict(quarantined_jobs=quarantined)),flush=True)
            active=[read(p) for p in (work/'workers').glob('*features*.json')]
            stale=[r['worker'] for r in active if r['status'] not in ('finished','failed') and time.time()-r['heartbeat']>300]
            if stale:print(json.dumps(dict(stale_workers=stale)),flush=True)
            if all(p['finished_jobs']==p['jobs'] for p in phases.values()):
                result=audit(work,a.processes);break
            time.sleep(30)
    else:result=status(a.work) if a.command=='status' else audit(a.work,a.processes)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
