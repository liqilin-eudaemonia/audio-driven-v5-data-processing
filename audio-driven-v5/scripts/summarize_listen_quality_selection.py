#!/usr/bin/env python3
"""Read-only dataset overview from the final quality selection; no new labels."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_listen_richness_pilot import display_name

FACE = [0, 128, 160, 192, 224, 256, 320, 384, 512, math.inf]
DURATION = [0, 3, 5, 7, 10, 15, math.inf]
RICHNESS = [0, 1, 2, 3, 4, 5, 6, 10, math.inf]
THRESHOLDS = [128, 160, 192, 224, 256, 320, 384, 512]


def read(p):
    return json.loads(Path(p).read_text())


def write_csv(p, rows, fields=None):
    rows = list(rows)
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def label(lo, hi):
    return f'>={lo:g}' if math.isinf(hi) else f'[{lo:g},{hi:g})'


def case(column, edges):
    clauses = [f'WHEN {column} IS NULL THEN -1']
    clauses += [f'WHEN {column} < {edge:g} THEN {i}' for i, edge in enumerate(edges[1:-1])]
    return 'CASE ' + ' '.join(clauses) + f' ELSE {len(edges)-2} END'


def distributions(db, column, edges, total_n, total_h):
    q = f'SELECT {case(column, edges)} AS bin,COUNT(*),SUM(duration)/3600 FROM samples GROUP BY bin'
    values = {i:(n,h) for i,n,h in db.execute(q)}
    result = []
    for i in [-1] + list(range(len(edges)-1)):
        n,h = values.get(i,(0,0.))
        result.append(dict(metric=column, bin='unknown' if i < 0 else label(edges[i],edges[i+1]),
                           clips=n, hours=h, clip_pct=100*n/total_n, hours_pct=100*h/total_h))
    assert sum(r['clips'] for r in result)==total_n
    assert math.isclose(sum(r['hours'] for r in result),total_h,abs_tol=1e-5)
    return result


def quantiles(db, column):
    rows = db.execute(f'SELECT {column},duration FROM samples WHERE {column} IS NOT NULL ORDER BY {column}').fetchall()
    def quantile(q):
        p=(len(rows)-1)*q; lo=math.floor(p); hi=math.ceil(p)
        return rows[lo][0]+(rows[hi][0]-rows[lo][0])*(p-lo)
    if not rows:return {'known_clips':0}
    weight=sum(r[1] for r in rows);acc=0.;weighted={}; targets=iter([.1,.5,.9]);current=next(targets,None)
    for value,duration in rows:
        acc+=duration
        while current is not None and acc>=weight*current:
            weighted[f'p{int(current*100)}']=value;current=next(targets,None)
    return dict(known_clips=len(rows),min=rows[0][0],max=rows[-1][0],
                mean=sum(r[0] for r in rows)/len(rows),
                clip_weighted={f'p{int(q*100)}':quantile(q) for q in (.05,.1,.25,.5,.75,.9,.95)},
                duration_weighted=weighted)


def run(work, output):
    work=Path(work).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    assert read(work/'complete.json')['passed'] and read(work/'audit.json')['passed']
    source=read(work/'summary.json');start=time.time()
    db=sqlite3.connect(f'file:{work}/selection.sqlite?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=MEMORY');db.execute('PRAGMA cache_size=-262144')
    n,h=db.execute('SELECT COUNT(*),SUM(duration)/3600 FROM samples').fetchone()
    assert n==source['counts']['retained'] and math.isclose(h,source['hours']['retained'],abs_tol=1e-5)
    split=[dict(split=s,clips=c,hours=t,clip_pct=100*c/n,hours_pct=100*t/h) for s,c,t in db.execute('SELECT split,COUNT(*),SUM(duration)/3600 FROM samples GROUP BY split')]
    families=[dict(source_family=s,clips=c,hours=t) for s,c,t in db.execute("SELECT CASE WHEN sid LIKE 'naturalistic_%' THEN 'naturalistic' WHEN sid LIKE 'improvised_%' THEN 'improvised' ELSE 'other' END AS family,COUNT(*),SUM(duration)/3600 FROM samples GROUP BY family")]
    assert not db.execute('SELECT group_id FROM samples GROUP BY group_id HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchall()
    assert not db.execute('SELECT pair FROM samples GROUP BY pair HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchall()
    result=dict(created_at=time.time(),source_work=str(work),clips=n,hours=h,split=split,source_families=families,
        source_pairs=db.execute('SELECT COUNT(DISTINCT pair) FROM samples').fetchone()[0],
        split_groups=db.execute('SELECT COUNT(DISTINCT group_id) FROM samples').fetchone()[0],
        mean_duration_s=h*3600/n,stat_status=dict(db.execute('SELECT status,COUNT(*) FROM samples GROUP BY status')))
    result['quantiles']={c:quantiles(db,c) for c in ('duration','face_p10','face_p50','types')}
    dist=[]
    for c,edges in [('duration',DURATION),('face_p10',FACE),('face_p50',FACE),('types',RICHNESS)]:
        dist+=distributions(db,c,edges,n,h)
    write_csv(output/'distributions.csv',dist);result['distributions']=dist
    face_split=[]
    for metric in ('face_p10','face_p50'):
        for split_name,bin_id,c,t in db.execute(f'SELECT split,{case(metric,FACE)} AS bin,COUNT(*),SUM(duration)/3600 FROM samples GROUP BY split,bin'):
            face_split.append(dict(split=split_name,metric=metric,bin='unknown' if bin_id<0 else label(FACE[bin_id],FACE[bin_id+1]),clips=c,hours=t))
    write_csv(output/'face_distribution_by_split.csv',face_split)
    thresholds=[]
    for metric in ('face_p10','face_p50'):
        for threshold in THRESHOLDS:
            c,t=db.execute(f'SELECT COUNT(*),COALESCE(SUM(duration)/3600,0) FROM samples WHERE {metric}>=?',(threshold,)).fetchone()
            tc,th=db.execute(f"SELECT COUNT(*),COALESCE(SUM(duration)/3600,0) FROM samples WHERE split='train' AND {metric}>=?",(threshold,)).fetchone()
            thresholds.append(dict(metric=metric,minimum_px=threshold,clips=c,hours=t,hours_pct=100*t/h,train_clips=tc,train_hours=th))
    write_csv(output/'face_thresholds.csv',thresholds);result['face_thresholds']=thresholds
    capacity=[]
    for threshold in (160,224,256,320,384):
        for minimum_types in (0,2,3,4,5):
            c,t=db.execute("SELECT COUNT(*),COALESCE(SUM(duration)/3600,0) FROM samples WHERE split='train' AND face_p10>=? AND types>=?",(threshold,minimum_types)).fetchone()
            capacity.append(dict(face_p10_min_px=threshold,adopted_types_min=minimum_types,train_clips=c,train_hours=t))
    write_csv(output/'training_pool_capacity.csv',capacity);result['training_pool_capacity']=capacity
    print(json.dumps({'stage':'basic_distributions_complete','elapsed_s':time.time()-start}),flush=True)
    # Per-class denominators are clips, never the sum of overlapping class durations.
    categories=list(csv.DictReader((work/'behavior_catalog.csv').open(encoding='utf-8-sig')))
    adopted=[];pending_only=[]
    for row in categories:
        key=row['type'];modality,name=key.split(':',1)
        r=dict(type=key,display_name=display_name(key),modality=modality,name=name,
            clips=int(row['sample_count']),hours=float(row['covered_clip_seconds'])/3600,
            source_pairs=int(row['pair_count']),pending_clips=int(row['pending_samples']),
            known_occurrence_clips=int(row['known_occurrence_samples']),unknown_occurrence_clips=int(row['unknown_occurrence_samples']),
            literal_occurrences=int(row['literal_occurrences_sum']) if row['literal_occurrences_sum'] else None,
            derived_occurrences=int(row['derived_occurrences_sum']) if row['derived_occurrences_sum'] else None,
            supported_occurrences=int(row['supported_occurrences_sum']) if row['supported_occurrences_sum'] else None,
            state_groups=int(row['state_groups']))
        r.update(clip_pct=100*r['clips']/n,hours_pct=100*r['hours']/h)
        (adopted if r['clips'] else pending_only).append(r)
    adopted.sort(key=lambda r:(-r['hours'],r['type']));by_type={r['type']:r for r in adopted}
    query='''SELECT b.type,COUNT(*),SUM(s.duration)/3600,
      SUM(CASE WHEN s.split='train' THEN 1 ELSE 0 END),SUM(CASE WHEN s.split='train' THEN s.duration ELSE 0 END)/3600,
      SUM(CASE WHEN s.split='dev' THEN 1 ELSE 0 END),SUM(CASE WHEN s.split='dev' THEN s.duration ELSE 0 END)/3600,
      SUM(CASE WHEN s.split='test' THEN 1 ELSE 0 END),SUM(CASE WHEN s.split='test' THEN s.duration ELSE 0 END)/3600,
      AVG(s.face_p10),AVG(s.face_p50)'''
    query+=''.join(f',SUM(CASE WHEN s.face_p10>={t} THEN s.duration ELSE 0 END)/3600' for t in (160,224,256,320,384,512))
    query+=' FROM behaviors b JOIN samples s ON b.sid=s.sid WHERE b.counted=1 GROUP BY b.type'
    seen=set()
    for values in db.execute(query):
        key,c,t,*tail=values;r=by_type[key];seen.add(key)
        assert c==r['clips'] and math.isclose(t,r['hours'],abs_tol=1e-5)
        names=['train_clips','train_hours','dev_clips','dev_hours','test_clips','test_hours','mean_face_p10_px','mean_face_p50_px']+[f'face_p10_ge_{v}_hours' for v in (160,224,256,320,384,512)]
        r.update(zip(names,tail))
        assert math.isclose(r['train_hours']+r['dev_hours']+r['test_hours'],r['hours'],abs_tol=1e-5)
    assert seen==set(by_type) and len(adopted)==source['distinct_candidate_names']
    for i,r in enumerate(adopted,1):r['rank_by_hours']=i
    write_csv(output/'all_categories.csv',adopted)
    write_csv(output/'pending_only_categories.csv',pending_only)
    result['category_name_counts']=dict(adopted=len(adopted),pending_only=len(pending_only),by_modality=dict(collections.Counter(r['modality'] for r in adopted)))
    result['category_rarity']=[dict(sample_count_range=name,category_names=sum(lo<=r['clips']<hi for r in adopted)) for name,lo,hi in [('1',1,2),('2-9',2,10),('10-99',10,100),('100-999',100,1000),('1000-9999',1000,10000),('>=10000',10000,math.inf)]]
    result['top_visual']=[r for r in adopted if r['modality']=='visual'][:30]
    result['all_audio']=[r for r in adopted if r['modality']=='audio']
    result['quantity_summary']=dict(db.execute("SELECT 'known_occurrence_records',COUNT(*) FROM behaviors WHERE counted=1 AND unit='occurrence' AND quantity IS NOT NULL UNION ALL SELECT 'unknown_occurrence_records',COUNT(*) FROM behaviors WHERE counted=1 AND unit='occurrence' AND quantity IS NULL UNION ALL SELECT 'supported_occurrences_sum',SUM(quantity) FROM behaviors WHERE counted=1 AND unit='occurrence' AND quantity IS NOT NULL UNION ALL SELECT 'state_groups',SUM(state) FROM behaviors WHERE counted=1"))
    # Sparse long tables: absent class/bin combinations represent zero clips.
    face_category=[]
    for metric in ('face_p10','face_p50'):
        q=f'''SELECT b.type,{case('s.'+metric,FACE)} AS bin,COUNT(*),SUM(s.duration)/3600
          FROM behaviors b JOIN samples s ON b.sid=s.sid WHERE b.counted=1 GROUP BY b.type,bin'''
        totals=collections.Counter()
        for key,bin_id,c,t in db.execute(q):
            face_category.append(dict(type=key,metric=metric,bin='unknown' if bin_id<0 else label(FACE[bin_id],FACE[bin_id+1]),clips=c,hours=t));totals[key]+=c
        assert all(totals[k]==r['clips'] for k,r in by_type.items())
    write_csv(output/'category_face_distribution.csv',face_category)
    db.execute('''CREATE TEMP TABLE modality AS SELECT sid,
      MAX(CASE WHEN counted=1 AND type LIKE 'visual:%' THEN 1 ELSE 0 END) AS visual,
      MAX(CASE WHEN counted=1 AND type LIKE 'audio:%' THEN 1 ELSE 0 END) AS audio
      FROM behaviors GROUP BY sid''')
    db.execute('CREATE INDEX temp.modality_id ON modality(sid)')
    result['modality_coverage']=[dict(visual=bool(v),audio=bool(a),clips=c,hours=t) for v,a,c,t in db.execute('''SELECT COALESCE(m.visual,0),COALESCE(m.audio,0),COUNT(*),SUM(s.duration)/3600
      FROM samples s LEFT JOIN modality m ON s.sid=m.sid GROUP BY COALESCE(m.visual,0),COALESCE(m.audio,0)''')]
    assert sum(r['clips'] for r in result['modality_coverage'])==n
    print(json.dumps({'stage':'category_tables_complete','elapsed_s':time.time()-start}),flush=True)
    # Sequential existing metadata read only; no raw waveform/video/measurement scan.
    kept={sid for (sid,) in db.execute('SELECT sid FROM samples')}
    audio=collections.Counter();nod=collections.Counter();crops=collections.Counter();support=collections.Counter();got=0
    raw_path=Path(source['source_work'])/'tables/sample_stats.jsonl'
    with raw_path.open() as f:
        for line in f:
            r=json.loads(line)
            if r['sample_id'] not in kept:continue
            kept.remove(r['sample_id']);got+=1;duration=r['duration_s'];a=r['audio'];nd=r['nod']
            audio['core_seconds']+=a['official_core_s'];audio['nominal_mask_seconds']+=a['nominal_keep_s']
            has_core=a['official_core_s']>0;silent=a['waveform_full_silence']
            audio['core_positive_clips']+=int(has_core);audio['core_positive_clip_seconds']+=duration*has_core
            audio['silent_clips']+=int(silent);audio['silent_clip_seconds']+=duration*silent
            audio['official_bouts']+=a['official_bout_count']
            nod['candidate_intervals']+=nd['candidate_interval_count'];nod['support_seconds']+=nd['support_s']
            nod['positive_clips']+=int(nd['candidate_interval_count']>0)
            nod['positive_clip_seconds']+=duration*int(nd['candidate_interval_count']>0)
            crops[r['crop_revision']]+=1
            if r['face_sample_detection_fraction'] is not None:
                support['face_detection_fraction_sum']+=r['face_sample_detection_fraction'];support['known_clips']+=1
            for key in ('face_short_side_p10','face_short_side_p50'):
                if r[key] is None:support[key+'_missing']+=1
    assert got==n and not kept
    result.update(audio_evidence=dict(audio),nod_evidence=dict(nod),crop_provenance=dict(crops),face_measurement_support=dict(support))
    result['adopted_behavior_clip_associations']=sum(r['clips'] for r in adopted)
    assert result['adopted_behavior_clip_associations']==db.execute('SELECT SUM(types) FROM samples').fetchone()[0]
    result['sum_class_coverage_hours']=sum(r['hours'] for r in adopted)
    result['definitions']=dict(category_names_are_open_not_exclusive_taxonomy=True,
        category_hours='sum of full clip duration once per adopted category per clip; categories overlap',
        face_size='native detected face short side, sampled at 5fps; per-clip p10 and p50, not 512px crop size',
        face_distribution_hours='full clip duration grouped by its face-size summary; not per-frame exposure duration',
        no_new_semantic_mapping=True,unknown_occurrence_count_is_null=True,
        all_zero_adopted_categories_not_verified_absence=True)
    result['seconds']=time.time()-start
    (output/'overview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix in ('.csv','.json') and p.name!='audit.json'}
    audit=dict(passed=True,at=time.time(),clips=n,hours=h,category_count=len(adopted),
       per_class_counts_and_hours_match_source=True,all_mutually_exclusive_distributions_sum_to_dataset=True,
       full_metadata_rows_matched=got,split_and_group_overlap=0,files=hashes)
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'completed':True,'clips':n,'hours':h,'category_names':result['category_name_counts'],'seconds':result['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.work,a.output)
