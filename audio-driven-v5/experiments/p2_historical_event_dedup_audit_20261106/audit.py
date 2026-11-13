"""Read-only source/fusion/temporal supervision audit; no relabeling or training."""
import sqlite3
import json
import sys
import random
import time
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.annotations import union, intersection_length

FUSION = ROOT/'experiments/listen_full_fusion_crop384_20261025'
CURRENT = ROOT/'experiments/p2_ltx_event_current_labels500h_v3_20261106/labels.sqlite'
OUT = Path(__file__).resolve().parent
VISUAL = ('head_nod','head_shake','smile','laugh','frown','surprise')


def audit_shard(path):
    path=Path(path);counts=Counter();issues=[];examples={};seconds=Counter()
    rows={r['sample_id']:r for r in (json.loads(l) for l in path.open())}
    evidence=FUSION/'evidence'/path.name
    for line in evidence.open():
        record=json.loads(line);sid=record['sample_id'];label=rows.pop(sid)
        raw={e['event_id']:e for e in record['original_events']}
        counts['samples']+=1
        complete=label['raw_completion_status']=='complete_candidates'
        counts['complete_candidates']+=complete
        counts['raw_visual_observations']+=sum(e['modality']=='visual' for e in raw.values())
        mapped={i for i,e in raw.items() if e['modality']=='visual' and e.get('mapping_status')=='mapped_candidate'}
        represented=set()
        by_class={c:[] for c in VISUAL}
        for e in label['events']:
            if e['modality']!='visual':continue
            counts['retained_visual_events']+=1
            members=[raw[i] for i in e['source_observation_ids']]
            represented.update(e['source_observation_ids'])
            for c in e['categories']:
                if c in by_class:by_class[c].append([e['start'],e['end']])
            if len(members)>1:
                counts['merged_visual_events']+=1
                counts['merged_visual_observations']+=len(members)
                if not all(x['categories']==e['categories'] for x in members):counts['merged_category_mismatches']+=1
                if not any((e['start'],e['end'])==(x['start'],x['end']) for x in members):counts['invented_representative_intervals']+=1
                if any(max(abs(a['start']-b['start']),abs(a['end']-b['end']))>.20000001 for a in members for b in members):counts['merged_boundary_spread_over_200ms']+=1
                counts['merged_distinct_description_variants']+=len(e.get('original_descriptions',[]))>1
                examples.setdefault('merged_visual',dict(sample_id=sid,integrated_event=e,original_members=members))
        counts['mapped_visual_observations']+=len(mapped)
        counts['mapped_visual_observations_missing_from_events']+=len(mapped-represented)
        for c,spans in by_class.items():
            counts['visual_intervals_before_union']+=len(spans)
            merged=union(spans);counts['visual_intervals_after_union']+=len(merged)
            seconds['summed_visual_event_seconds']+=sum(b-a for a,b in spans)
            seconds['class_union_visual_seconds']+=sum(b-a for a,b in merged)
            if union(spans+spans)!=merged:counts['duplicate_invariance_failures']+=1
        pending=[e for e in raw.values() if e['modality']=='visual' and e.get('mapping_status')=='needs_review']
        counts['unresolved_visual_observations']+=len(pending)
        counts['unresolved_visual_observations_with_candidate_category']+=sum(bool(e.get('categories')) for e in pending)
        counts['samples_with_unresolved_visual']+=bool(pending)
        counts['complete_samples_with_unresolved_visual']+=bool(pending) and complete
        if pending and complete:
            compound=[e for e in pending if 'multiple' in e.get('semantic_reason','').lower()]
            if compound:
                counts['complete_samples_with_explicit_multiple_action_uncertainty']+=1
                examples.setdefault('multiple_action_uncertainty',dict(sample_id=sid,source=label['source'],events=compound[:2],completion=label['raw_completion_status']))
            categorized=[e for e in pending if e.get('categories')]
            if categorized and 'uncertain_candidate_category' not in examples:
                examples['uncertain_candidate_category']=dict(sample_id=sid,events=categorized[:2])
            if len(issues)<12 or (categorized and not any(x['pending'].get('categories') for x in issues)):
                chosen=(categorized or compound or pending)[0]
                issues.append(dict(sample_id=sid,source=label['source'],duration_s=label['duration_s'],pending=chosen,explicit_multiple_action_uncertainty=bool(compound)))
    if rows:raise ValueError('Missing evidence rows')
    return dict(shard=str(path),counts=dict(counts),seconds=dict(seconds),issues=issues,examples=examples)


def trace_current(candidates):
    by_id={x['sample_id']:x for x in candidates};matched=[];risky=[];count=0
    db=sqlite3.connect(CURRENT.as_uri()+'?mode=ro',uri=True)
    for ordinal,sid,origin_text in db.execute("SELECT ordinal,sid,json_extract(value,'$.origins') FROM labels ORDER BY ordinal"):
        count+=1
        origins=json.loads(origin_text)
        for oldsid in set(origins)&by_id.keys():
            item=by_id[oldsid]
            if len(matched)<12:matched.append(dict(current_sample_id=sid,historical_sample_id=oldsid))
            raw=db.execute('SELECT value FROM labels WHERE ordinal=?',(ordinal,)).fetchone()[0];current=json.loads(raw)
            # Current start is obtained from the frozen CURRENT source rather than
            # inferred from a label's event intervals or its original cut.
            if not hasattr(trace_current,'source_db'):
                trace_current.source_db=sqlite3.connect((ROOT/'experiments/p2_ltx_event_current_sources_v2_20261106/sources.sqlite').as_uri()+'?mode=ro',uri=True)
            source=json.loads(trace_current.source_db.execute('SELECT value FROM samples WHERE sid=?',(sid,)).fetchone()[0])
            current_start=source['source']['source_start_s'];old_start=item['source']['source_start_s']
            e=item['pending'];a=max(0.,old_start+e['start']-current_start);b=min(current['duration_s'],old_start+e['end']-current_start)
            if b<=a:continue
            selected_categories=item['pending'].get('categories') or VISUAL
            for c in selected_categories:
                fields=current['classes'][c]
                for i in range(1,int(current['duration_s']/.32)+2):
                    left=.04+(i-1)*.32;right=.04+i*.32
                    if right>current['duration_s']+1e-6:continue
                    if left>=a-1e-6 and right<=b+1e-6 and intersection_length(fields['coverage'],left,right)>=right-left-1e-6 and intersection_length(fields['unknown'],left,right)<=1e-8 and intersection_length(fields['spans'],left,right)<=1e-8:
                        risky.append(dict(current_sample_id=sid,historical_sample_id=oldsid,category=c,current_bin=[left,right],numeric_target=0,background_weight=.1,historical_pending=e,historical_interval_on_current_clip=[a,b],explicit_multiple_action_uncertainty=item['explicit_multiple_action_uncertainty']))
                        break
                if len(risky)>=8 and any(x['explicit_multiple_action_uncertainty'] for x in risky):break
            if len(risky)>=8 and any(x['historical_pending'].get('categories') for x in risky):break
        if len(risky)>=8 and any(x['historical_pending'].get('categories') for x in risky):break
    db.close()
    return dict(scanned_current_rows=count,candidate_old_rows=len(by_id),matches=matched,weak_background_on_unresolved_examples=risky,scope='Targeted trace of sampled unresolved historical candidates, not a prevalence estimate')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
    started=time.time();all_paths=sorted((FUSION/'labels').glob('*.jsonl'));paths=all_paths if args.all else random.Random(20261106).sample(all_paths,32)
    counts=Counter();seconds=Counter();issues=[];examples={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(audit_shard,paths),1):
            counts.update(r['counts']);seconds.update(r['seconds']);issues+=r['issues']
            for k,v in r['examples'].items():examples.setdefault(k,v)
            if i%25==0 or i==len(paths):print(json.dumps(dict(stage='source_audit',completed_shards=i,total_shards=len(paths),samples=counts['samples'],elapsed_s=round(time.time()-started,1))),flush=True)
    # Prioritize compound observations so that the trace checks the specific
    # multi-action limitation rather than only unrelated out-of-scope actions.
    issues.sort(key=lambda x:(not bool(x['pending'].get('categories')),not x['explicit_multiple_action_uncertainty']))
    trace=trace_current(issues)
    result=dict(complete=True,read_only=True,seed=20261106,source_total_shards=len(all_paths),sampled_shards=[str(p) for p in paths],counts=dict(counts),seconds=dict(seconds),examples=examples,current_trace=trace,elapsed_s=time.time()-started)
    result['scope']='all historical fusion rows' if args.all else '32 deterministically sampled historical shards'
    output=OUT/('AUDIT_FULL.json' if args.all else 'AUDIT.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sampled_shards','examples','current_trace')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
