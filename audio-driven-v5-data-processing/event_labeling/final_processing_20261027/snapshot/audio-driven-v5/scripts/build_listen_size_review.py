"""Sample existing final candidates and publish native-face/crop review; no re-screening."""
import collections
import concurrent.futures
import copy
import hashlib
import json
import random
import shutil
from pathlib import Path
import export_annotation_sample50 as media

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data_previews/seamless_listen_size_review'
FULL = ROOT/'data_corpora/seamless_listen_local_full'
RANDOM = ROOT/'data_exports/seamless_annotation_sample50_20261023'
FACE = [0,64,96,128,160,192,256,320,448,float('inf')]
CROP = [0,256,320,384,448,512,640,768,float('inf')]


def group(value, edges, prefix):
    if value is None:return prefix+'_missing'
    return next(prefix+'_'+str(i) for i,(a,b) in enumerate(zip(edges,edges[1:])) if a<=value<b)


def memberships(row):
    face=row['face_geometry']['listener'];crop=row['crop_proposals']['listener']
    return [group(face.get('face_short_side_px',{}).get('p10'),FACE,'face'),
            group(crop.get('size'),CROP,'crop')]


def selection():
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    rng=random.Random(2026102401);counts=collections.Counter();reservoir={}
    with media.SOURCE.open() as f:
        for i,line in enumerate(f,1):
            row=json.loads(line)
            for key in memberships(row):
                counts[key]+=1;limit=3 if key.startswith('face') else 2
                bucket=reservoir.setdefault(key,[])
                if len(bucket)<limit:bucket.append(row)
                else:
                    slot=rng.randrange(counts[key])
                    if slot<limit:bucket[slot]=row
            if i%200000==0:print('Scanned',i,flush=True)
    unique={r['sample_id']:r for rows in reservoir.values() for r in rows}
    manifest_index=json.loads((FULL/'manifests/index.json').read_text())
    result=dict(seed=2026102401,population=i,group_counts=dict(counts),
                source_manifest_sha256=manifest_index['manifest_sha256']['train_primary_keep_candidate'],
                sampled_groups={k:[r['sample_id'] for r in v] for k,v in reservoir.items()},
                rows=list(unique.values()),method='Uniform reservoir per face/crop stratum; sample counts are not population proportions.')
    media.write(path,result)
    return result


def prefixed(row,prefix):
    r=copy.deepcopy(row)
    for v in r['roles'].values():
        for k in ('clip_video','context_video','audio_wav'):v[k]=prefix+'/'+v[k]
    for k in ('metadata_file','quality_evidence_file'):r[k]=prefix+'/'+r[k]
    return r


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=selection()
    assert selected['population']==1200212
    link=OUT/'random'
    if not link.exists():link.symlink_to(RANDOM,target_is_directory=True)
    random_rows=json.loads((RANDOM/'items.json').read_text())
    items={r['sample_id']:dict(prefixed(r,'random'),cohorts=['random']) for r in random_rows}
    media.OUT=OUT/'size_samples';media.OUT.mkdir(exist_ok=True)
    def build(task):
        i,r=task
        cached=media.OUT/'samples'/f"{i:03d}_{r['sample_id']}"/'annotation_item.json'
        if cached.exists():return json.loads(cached.read_text())
        return media.export(task)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(build,enumerate([r for r in selected['rows'] if r['sample_id'] not in items],1)):
            items[r['sample_id']]=dict(prefixed(r,'size_samples'),cohorts=[])
    for key,ids in selected['sampled_groups'].items():
        for ident in ids:items[ident]['cohorts'].append(key)
    groups=[dict(id='random',name='随机主池样本 · 50 条',count=50)]
    for prefix,edges,label in [('face',FACE,'人脸短边 p10'),('crop',CROP,'参考裁框边长')]:
        for i,(a,b) in enumerate(zip(edges,edges[1:])):
            key=prefix+'_'+str(i);name=f'{label} {a}–{b:g}px' if b!=float('inf') else f'{label} ≥{a}px'
            groups.append(dict(id=key,name=name,count=selected['group_counts'].get(key,0)))
        key=prefix+'_missing'
        groups.append(dict(id=key,name=label+' · 未测量',count=selected['group_counts'].get(key,0)))
    face=json.loads((FULL/'face_size_distribution.json').read_text())
    crop=json.loads((FULL/'crop_size_distribution.json').read_text())
    summary=json.loads((FULL/'screening_summary.json').read_text())
    key='train/primary/keep_candidate'
    result=dict(revision='seamless-face-size-review-20261024',groups=groups,items=list(items.values()),
                statistics=dict(utc=summary['utc'],main_hours=summary['hours'][key],
                    face=face['pools'][key]['roles']['listener'],crop=crop['pools'][key]['roles']['listener'],
                    pixel_edges=crop['pixel_edges']),visual_filter_applied=False)
    result['statistics_by_pool']={'primary':result['statistics']}
    addition=OUT/'reserve_addition.json'
    if addition.exists():
        reserve=json.loads(addition.read_text())
        result['groups']+=reserve['groups'];result['items']+=reserve['items']
        result['statistics_by_pool']['onset_reserve']=reserve['statistics']
        result['onset_revision']=reserve['revision']
    accepted=FULL/'accepted_audio_candidates/index.json'
    if accepted.exists():
        result['accepted_audio_candidates']=json.loads(accepted.read_text())
    for r in result['items']:
        for v in r['roles'].values():
            for k in ('clip_video','context_video','audio_wav'):assert (OUT/v[k]).is_file()
    media.write(OUT/'review.json',result)
    shutil.copy2(ROOT/'scripts/listen_size_review.html',OUT/'index.html')
    # Chinese glyphs for browser environments without system CJK fonts.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=(OUT/'index.html').read_text()+(OUT/'review.json').read_text());sub.subset(font)
    font.flavor='woff';font.save(OUT/'review.woff')
    web=ROOT/'data_previews/seamless_face_listening_v1/listen_size_review'
    if not web.exists():web.symlink_to(OUT,target_is_directory=True)
    media.write(OUT/'build_validation.json',dict(samples=len(result['items']),random=50,
        stratified_unique=len(items)-50,onset_reserve=len(result['items'])-len(items),all_media_paths_exist=True,visual_filter_applied=False,
        source_stats_sha256={p:hashlib.sha256((FULL/p).read_bytes()).hexdigest() for p in
                            ['face_size_distribution.json','crop_size_distribution.json','screening_summary.json']}))
    print('READY http://127.0.0.1:8875/listen_size_review/',len(result['items']),flush=True)


if __name__=='__main__':main()
