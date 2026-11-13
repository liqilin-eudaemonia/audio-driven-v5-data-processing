#!/usr/bin/env python3
"""Read-only corpus report: open candidates, canonical labels, and unknowns.

Keeps every authorized ID, including empty-event and unadmitted samples.
Never converts unknown observations to negative training labels.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_listen.contracts import atomic_json,digest,sha
from listen_benchmark.backchannel_face_stats import loads,CATEGORIES


def merge(spans):
    result=[]
    for a,b in sorted(spans):
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def seconds(spans):return sum(b-a for a,b in merge(spans))


def event_stats(events,duration,canonical=False):
    units=set();categories=[set() for _ in CATEGORIES];spans=[[] for _ in CATEGORIES]
    visual=set();audio=set();vs=[];aus=[];all_spans=[];uncategorized=set();invalid=0
    for event in events:
        key=(event.get('source',{}).get('event_id') or event['event_id']) if canonical else event['event_id']
        cats=[event['type']] if canonical else event.get('categories',[])
        modality=event.get('source',{}).get('modality') if canonical else event.get('modality')
        units.add(key)
        if modality=='visual':visual.add(key)
        elif modality=='audio':audio.add(key)
        selected=[CATEGORIES.index(c) for c in set(cats) if c in CATEGORIES]
        if not selected:uncategorized.add(key)
        for i in selected:categories[i].add(key)
        support=[dict(start=event.get('start'),end=event.get('end'))] if canonical else event.get('intervals',[dict(start=event.get('start'),end=event.get('end'))])
        for x in support:
            a,b=x.get('start'),x.get('end')
            if not all(type(t) in (int,float) and math.isfinite(t) for t in (a,b)) or not 0<=a<b<=duration+.0011:
                invalid+=1;continue
            b=min(b,duration);all_spans.append((a,b))
            if modality=='visual':vs.append((a,b))
            elif modality=='audio':aus.append((a,b))
            for i in selected:spans[i].append((a,b))
    return dict(events=len(units),interval_entries=len(events),visual=len(visual),audio=len(audio),uncategorized=len(uncategorized),
        counts=[len(v) for v in categories],class_seconds=[seconds(v) for v in spans],
        mask=sum(1<<i for i,v in enumerate(categories) if v),visual_seconds=seconds(vs),audio_seconds=seconds(aus),
        event_seconds=seconds(all_spans),overlap_seconds=max(0.,seconds(vs)+seconds(aus)-seconds(vs+aus)),invalid_intervals=invalid)


def process(args):
    root,out,entry=args;root=Path(root);out=Path(out);taskid=entry['task_id'];dest=out/'shards'/f'{taskid}.jsonl';marker=dest.with_suffix('.done.json')
    ready=loads((root/'ready'/f'{taskid}.json').read_bytes())
    if marker.exists():
        m=loads(marker.read_bytes())
        if m['ready_sha256']!=sha(root/'ready'/f'{taskid}.json') or m['output_sha256']!=sha(dest):raise ValueError('Report resume identity differs')
        return m
    old=Path(entry['path']).read_bytes()
    if hashlib.sha256(old).hexdigest()!=entry['sha256']:raise ValueError('Final audit changed')
    audited=loads(old);jobs_body=Path(ready['path']).read_bytes();opened_body=Path(ready['integrated']).read_bytes()
    if hashlib.sha256(jobs_body).hexdigest()!=ready['sha256'] or hashlib.sha256(opened_body).hexdigest()!=ready['integrated_sha256']:raise ValueError('Source shard changed')
    jobs=[loads(line) for line in jobs_body.splitlines()];opened=[loads(line) for line in opened_body.splitlines()]
    if len(jobs)!=len(opened) or len(jobs)!=ready['samples']:raise ValueError('Task count mismatch')
    rows=[]
    for job,label in zip(jobs,opened):
        obs=job['observation'];sid=obs['sample_id'];duration=obs['requested_duration_s']
        canonical=loads((root/'view/labels/canonical'/f'{sid}.json').read_bytes())
        if digest(canonical)!=job['canonical_sha256']:raise ValueError('Canonical label changed')
        if sid!=label['sample_id'] or sid!=canonical['sample_id'] or obs['official_split']!='train':raise ValueError('Scope identity differs')
        op=event_stats(label['events'],duration);ca=event_stats(canonical['events'],duration,True)
        vc=all(bool(canonical['coverage'][c]) for c in CATEGORIES[:-1]);ac=bool(canonical['coverage'][CATEGORIES[-1]])
        r=dict(sample_id=sid,corpus=sid.split('_')[0],seconds=duration,training_admitted=int(canonical['training_admitted']),
            full_visual_receipt=int(vc),full_audio_receipt=int(ac),
            raw_issue_count=label.get('raw_issue_count',0),semantic_issue_count=label.get('semantic_issue_count',0),
            unresolved_observations=len(label.get('unresolved_observations',[])),
            out_of_scope_observations=sum(v.get('reason')=='out_of_scope' for v in label.get('unresolved_observations',[])),
            needs_review_observations=sum(v.get('reason')!='out_of_scope' for v in label.get('unresolved_observations',[])),
            unknown_interval_classes=sum(bool(v) for v in canonical['unknown'].values()))
        for prefix,stats in [('open',op),('canonical',ca)]:
            r.update({prefix+'_'+k:v for k,v in stats.items() if k not in ('counts','class_seconds')})
            for i,c in enumerate(CATEGORIES):r[prefix+'_'+c+'_count']=stats['counts'][i];r[prefix+'_'+c+'_seconds']=stats['class_seconds'][i]
        # out_of_scope (e.g. gaze/one head turn) is not a failed annotation.
        r['quiet_candidate']=int(op['events']==0 and vc and ac and not r['raw_issue_count'] and not r['semantic_issue_count'] and not r['needs_review_observations'] and not r['unknown_interval_classes'])
        rows.append(r)
    if {r['sample_id'] for r in rows}!=set(audited['ids']):raise ValueError('Final audited ID set changed')
    with dest.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    result=dict(task_id=taskid,samples=len(rows),output_sha256=sha(dest),ready_sha256=sha(root/'ready'/f'{taskid}.json'))
    atomic_json(marker,result);return result


def aggregate(root,out,audit):
    import numpy as np
    sums=Counter();class_samples={p:Counter() for p in ['open','canonical']};class_events={p:Counter() for p in ['open','canonical']}
    class_seconds={p:Counter() for p in ['open','canonical']};class_clip_seconds={p:Counter() for p in ['open','canonical']}
    combinations={p:Counter() for p in ['open','canonical']};modality={p:Counter() for p in ['open','canonical']}
    buckets=Counter();bucket_seconds=Counter();corpus={};hist=Counter();cardinality=Counter();cooccur=[[0]*7 for _ in range(7)]
    quant={k:[] for k in ['seconds','open_events','canonical_events','open_audio','open_visual']}
    seen=set();first=True;db=sqlite3.connect(out/'sample_index.sqlite');db.execute('CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,open_events INT,canonical_events INT,open_mask INT,canonical_mask INT,training_admitted INT,quiet_candidate INT,seconds REAL)');db.execute('DELETE FROM samples');db.commit()
    with (out/'samples.csv').open('w',newline='') as stream,(out/'sample_ids_all.txt').open('w') as all_ids,(out/'no_open_event_ids.txt').open('w') as empty_ids,(out/'unadmitted_ids.txt').open('w') as unadmitted_ids:
        for entry in audit['tasks']:
            rows=[loads(line) for line in (out/'shards'/f'{entry["task_id"]}.jsonl').open('rb')]
            if first:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();first=False
            writer.writerows(rows)
            for r in rows:
                sid=r['sample_id']
                if sid in seen:raise ValueError('Duplicate report ID')
                seen.add(sid);all_ids.write(sid+'\n');sums['samples']+=1;sums['seconds']+=r['seconds'];sums['training_admitted']+=r['training_admitted']
                for k in ['raw_issue_count','semantic_issue_count','unresolved_observations','out_of_scope_observations','needs_review_observations']:sums[k]+=r[k];sums['samples_with_'+k]+=r[k]>0
                sums['quiet_candidate']+=r['quiet_candidate'];sums['full_visual_receipt']+=r['full_visual_receipt'];sums['full_audio_receipt']+=r['full_audio_receipt'];sums['both_full_receipts']+=r['full_visual_receipt'] and r['full_audio_receipt']
                if not r['training_admitted']:unadmitted_ids.write(sid+'\n')
                if r['open_events']==0:empty_ids.write(sid+'\n')
                for key in quant:quant[key].append(r[key])
                group=corpus.setdefault(r['corpus'],Counter());group['samples']+=1;group['seconds']+=r['seconds'];group['open_events']+=r['open_events'];group['with_open_events']+=r['open_events']>0
                for p in ['open','canonical']:
                    sums[p+'_events']+=r[p+'_events'];sums[p+'_audio_events']+=r[p+'_audio'];sums[p+'_visual_events']+=r[p+'_visual'];sums[p+'_interval_entries']+=r[p+'_interval_entries'];sums[p+'_uncategorized_events']+=r[p+'_uncategorized'];sums[p+'_event_seconds']+=r[p+'_event_seconds'];sums[p+'_invalid_intervals']+=r[p+'_invalid_intervals']
                    sums[p+'_with_events']+=r[p+'_events']>0;sums[p+'_with_seven_class_events']+=r[p+'_mask']>0;sums[p+'_with_temporal_av_overlap']+=r[p+'_overlap_seconds']>1e-9;sums[p+'_temporal_av_overlap_seconds']+=r[p+'_overlap_seconds']
                    combinations[p][r[p+'_mask']]+=1
                    mode='both' if r[p+'_visual'] and r[p+'_audio'] else 'visual_only' if r[p+'_visual'] else 'audio_only' if r[p+'_audio'] else 'neither'
                    modality[p][mode]+=1
                    for i,c in enumerate(CATEGORIES):
                        n=r[p+'_'+c+'_count'];class_samples[p][c]+=n>0;class_events[p][c]+=n;class_seconds[p][c]+=r[p+'_'+c+'_seconds'];class_clip_seconds[p][c]+=r['seconds'] if n else 0
                mask=r['open_mask'];cardinality[mask.bit_count()]+=1
                for i in range(7):
                    for j in range(7):cooccur[i][j]+=bool(mask & (1<<i)) and bool(mask & (1<<j))
                count=r['open_events'];hist['0' if count==0 else '1' if count==1 else '2–5' if count<=5 else '6–10' if count<=10 else '11–20' if count<=20 else '>20']+=1
                bucket='with_seven_class_events' if mask else 'only_uncategorized_events' if count else 'quiet_candidate' if r['quiet_candidate'] else 'no_events_insufficient_evidence'
                buckets[bucket]+=1;bucket_seconds[bucket]+=r['seconds']
            db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',[(r['sample_id'],r['open_events'],r['canonical_events'],r['open_mask'],r['canonical_mask'],r['training_admitted'],r['quiet_candidate'],r['seconds']) for r in rows])
        db.commit()
    scope=sqlite3.connect(f'file:{root}/scope.sqlite?mode=ro',uri=True);expected={x[0] for x in scope.execute('SELECT sid FROM samples')};scope.close();db.close()
    if seen!=expected:raise ValueError('Missing report IDs')
    if dict(class_samples['canonical'])!=audit['class_sample_counts']:raise ValueError('Canonical prevalence disagrees with full audit')
    if sums['training_admitted']!=audit['counts']['training_admitted'] or sums['canonical_audio_events']!=audit['counts']['retained_audio_events'] or sums['open_audio_events']!=audit['counts']['open_retained_audio_events']:raise ValueError('Audio/admission counts differ from audited full output')
    summary=dict(scope=dict(samples=sums['samples'],seconds=sums['seconds'],hours=sums['seconds']/3600,split='train',listener_crop_floor=384),
        totals=dict(sums),classes={p:{c:dict(samples=class_samples[p][c],prevalence_percent=100*class_samples[p][c]/sums['samples'],events=class_events[p][c],union_seconds=class_seconds[p][c],containing_clip_hours=class_clip_seconds[p][c]/3600) for c in CATEGORIES} for p in ['open','canonical']},
        modality={p:dict(v) for p,v in modality.items()},buckets={k:dict(samples=v,hours=bucket_seconds[k]/3600,percent=100*v/sums['samples']) for k,v in buckets.items()},
        event_count_histogram=dict(hist),class_count_histogram=dict(cardinality),cooccurrence=cooccur,
        combinations={p:[dict(mask=k,classes=[c for i,c in enumerate(CATEGORIES) if k&(1<<i)],samples=v,percent=100*v/sums['samples']) for k,v in sorted(values.items(),key=lambda x:(-x[1],x[0]))] for p,values in combinations.items()},
        quantiles={k:{str(q):float(np.percentile(v,q)) for q in [0,10,25,50,75,90,95,99,100]} for k,v in quant.items()},
        by_corpus={k:dict(v) for k,v in corpus.items()},audio_postprocess_audit=audit['counts'],retained_all_samples=True,
        annotation_limit='Model candidate counts, not human-ground-truth occurrence counts. Missing labels are not confirmed absence.',
        training_flags_changed=False,source_root=str(root),source_audit_sha256=sha(root/'audit.json'),code_sha256=sha(__file__))
    atomic_json(out/'summary.json',summary)
    with (out/'class_statistics.csv').open('w',newline='') as f:
        fields=['scope','class','samples','prevalence_percent','events','union_seconds','containing_clip_hours'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in ['open','canonical']:
            for c in CATEGORIES:writer.writerow(dict(scope=p,**{'class':c},**summary['classes'][p][c]))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    root=Path(a.run).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True)
    complete=loads((root/'COMPLETE.json').read_bytes());audit=loads((root/'audit.json').read_bytes())
    assert complete['status']=='passed' and sha(root/'audit.json')==complete['audit_sha256']
    signature=dict(version='full-postprocessed-backchannel-report-v1',code_sha256=sha(__file__),source_completion_sha256=sha(root/'COMPLETE.json'))
    if (out/'provenance.json').exists() and loads((out/'provenance.json').read_bytes())!=signature:raise ValueError('Use a new output for changed code/scope')
    atomic_json(out/'provenance.json',signature);done=0;started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(process,((str(root),str(out),t) for t in audit['tasks']),chunksize=8),1):
            done+=result['samples']
            if i%256==0:
                progress=dict(shards=i,total_shards=len(audit['tasks']),samples=done,total_samples=complete['samples'],seconds=time.time()-started)
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=aggregate(root,out,audit)
    files=['summary.json','samples.csv','class_statistics.csv','sample_ids_all.txt','no_open_event_ids.txt','unadmitted_ids.txt','sample_index.sqlite','provenance.json']
    atomic_json(out/'COMPLETE.json',dict(status='passed',samples=done,files={f:sha(out/f) for f in files},source_audit_reconciled=True,all_ids_preserved=True))
    atomic_json(out/'progress.json',dict(phase='complete',samples=done,total_samples=complete['samples']))
    print(json.dumps(dict(status='passed',totals=summary['totals'],buckets=summary['buckets'])),flush=True)


if __name__=='__main__':main()
