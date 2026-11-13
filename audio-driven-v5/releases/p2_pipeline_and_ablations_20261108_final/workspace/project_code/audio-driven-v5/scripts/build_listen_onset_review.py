"""Read-only sampling of the audio-passed onset reserve for the existing review UI."""
import collections
import concurrent.futures
import hashlib
import json
import random
import tarfile
from pathlib import Path
import export_annotation_sample50 as media
import build_listen_size_review as review

ROOT=review.ROOT
OUT=review.OUT/'onset_samples'
SOURCE=review.FULL/'manifests/train_onset_reserve_keep_candidate.jsonl'


def select():
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    rng=random.Random(2026102402);counts=collections.Counter();hours=collections.Counter();buckets={}
    for line in SOURCE.open():
        r=json.loads(line)
        assert r['admission_pool']=='onset_reserve' and r['audio_final']['pool']=='keep_candidate'
        assert r['boundary_activity']['onset_activity']
        kind='crossing' if r['boundary_activity']['onset_crossing'] else 'early'
        for key,n in [('random',24),(kind,4)]:
            counts[key]+=1;hours[key]+=r['duration_s']/3600
            bucket=buckets.setdefault(key,[])
            if len(bucket)<n:bucket.append(r)
            else:
                i=rng.randrange(counts[key])
                if i<n:bucket[i]=r
    index=json.loads((review.FULL/'manifests/index.json').read_text())
    data=dict(seed=2026102402,population=dict(counts),hours=dict(hours),
              source_manifest_sha256=index['manifest_sha256']['train_onset_reserve_keep_candidate'],
              sampled_groups={k:[r['sample_id'] for r in v] for k,v in buckets.items()},
              rows=list({r['sample_id']:r for v in buckets.values() for r in v}.values()),
              method='24 uniform clip samples plus 4 within each onset type; supplements do not represent population proportions.')
    assert counts['random']==218912
    media.write(path,data)
    print(json.dumps(dict(population=dict(counts),hours=dict(hours),selected=len(data['rows']))),flush=True)
    return data


def build(task):
    i,r=task
    folder=OUT/'samples'/f"{i:03d}_{r['sample_id']}"
    cached=folder/'annotation_item.json'
    item=json.loads(cached.read_text()) if cached.exists() else media.export(task)
    source=r['sources']['listener'];member=str(Path(source['member']).with_suffix('.json'))
    with tarfile.open(source['archive'],'r:') as tar:
        raw=tar.extractfile(member).read()
    annotation=json.loads(raw)
    vad=sorted((float(v['start']),float(v['end'])) for v in annotation['metadata:vad'] if v['end']>v['start'])
    merged=[]
    for a,b in vad:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    near=any(a<r['start']+.5 and b>r['start'] for a,b in merged)
    crossing=any(a<r['start']-1e-7 and b>r['start']+1e-7 for a,b in merged)
    assert near and crossing==r['boundary_activity']['onset_crossing'],r['sample_id']
    evidence=dict(boundary_activity=r['boundary_activity'],
        listener_vad_source=[v for v in merged if v[1]>item['context_source_start_s'] and v[0]<item['context_source_end_s']],
        listener_events=r['listener_events'],source_annotation=dict(archive=source['archive'],member=member,sha256=hashlib.sha256(raw).hexdigest()),
        annotation_kind='Original dataset VAD; not new inference or confirmed backchannel',
        admission_rule_reproduced=True)
    media.write(folder/'onset_evidence.json',evidence)
    item['onset_evidence']=evidence
    media.write(cached,item)
    return item


def main():
    OUT.mkdir(parents=True,exist_ok=True);selected=select();media.OUT=OUT
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:items=list(pool.map(build,enumerate(selected['rows'],1)))
    for r in items:
        r['cohorts']=['reserve_all']+['reserve_'+k for k,ids in selected['sampled_groups'].items() if r['sample_id'] in ids]
    groups=[dict(id='reserve_all',name='起始活动储备池 · 全部抽样',count=218912,pool='onset_reserve',sampling='combined')]
    for key,label in [('random','均匀随机 24 条'),('crossing','补充：人声跨过切入点'),('early','补充：切入后 0.5 秒内起声')]:
        groups.append(dict(id='reserve_'+key,name='储备池 · '+label,count=selected['population'][key],pool='onset_reserve',sampling='random' if key=='random' else 'case'))
    face=json.loads((review.FULL/'face_size_distribution.json').read_text())
    crop=json.loads((review.FULL/'crop_size_distribution.json').read_text())
    summary=json.loads((review.FULL/'screening_summary.json').read_text());key='train/onset_reserve/keep_candidate'
    stats=dict(utc=summary['utc'],main_hours=summary['hours'][key],face=face['pools'][key]['roles']['listener'],crop=crop['pools'][key]['roles']['listener'],pixel_edges=crop['pixel_edges'])
    addition=dict(revision='onset-reserve-review-20261024',groups=groups,items=[review.prefixed(r,'onset_samples') for r in items],statistics=stats)
    media.write(review.OUT/'reserve_addition.json',addition)
    review.main()
    media.write(OUT/'validation.json',dict(samples=len(items),random=24,source_vad_verified=True,
        videos=len(items)*4,media_decoded=True,source_dimensions_preserved=True,policy_changed=False,
        population=selected['population'],hours=selected['hours']))
    print('READY http://127.0.0.1:8875/listen_size_review/?group=reserve_all',flush=True)


if __name__=='__main__':main()
