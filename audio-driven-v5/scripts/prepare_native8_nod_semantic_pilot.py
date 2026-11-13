#!/usr/bin/env python3
"""Light preprocessing only: source-time alignment and exact duplicate collapse."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_prompt import digest,source_payload,plan_schema,PLAN_PROMPT,REVIEW_PROMPT
from listen_visual_native8_nod import save,sha

SOURCE=ROOT/'experiments/listen_visual_native8_nod_20261029/bundles.jsonl'
OUT=ROOT/'experiments/listen_native8_nod_semantic15_20261029'

def prepare(bundle, *, deduplicate_exact=True, validate_plan=True):
    duration=bundle['duration_s'];original=[];excluded=[]
    for r in bundle['vlm_observations']:
        if r['status']!='valid':
            excluded.append({'view':r['view'],'status':r['status'],'raw':r['raw'],'reason':'invalid response retained outside LLM input'})
            continue
        for i,e in enumerate(r['parsed']['events']):
            a,b=r['start']+e['start'],r['start']+e['end']
            if not 0<=a<b<=duration+1e-8:raise ValueError('Bad source interval')
            original.append({'observation_id':f"vlm-{r['view']}-{i+1:03d}",'start':a,'end':b,
                'description':e['behavior'],'modality':'visual','source_type':'vlm_video8',
                'view':r['view'],'window_start':r['start'],'local_event':e,
                'response_sha256':digest(r)})
    for e in bundle['nod_candidates']:
        original.append({'observation_id':e['event_id'],'start':e['start'],'end':e['end'],
                         'description':e['description'],'modality':'visual','source_type':'glare_nod_recall',
                         'detector_event':e})
    groups={};events=[]
    for observation in original:
        # 1 ns comparison precision only removes float addition noise; kept bounds
        # and source-local bounds are not changed. Never collapse different sources.
        key=(observation['source_type'],round(observation['start'],9),round(observation['end'],9),observation['description'])
        if deduplicate_exact and key in groups:
            groups[key]['original_observations'].append(observation)
            continue
        event={'event_id':f'obs-{len(events)+1:04d}','start':observation['start'],'end':observation['end'],
               'modality':'visual','description':observation['description'],'source_type':observation['source_type'],
               'original_observations':[observation]}
        groups[key]=event;events.append(event)
    record={'sample_id':bundle['sample_id'],'duration_s':duration,'events':events,
            'preprocessing':{'original_observations':len(original),'exact_duplicates_collapsed':len(original)-len(events),
                'invalid_responses':excluded,'time_rule':'window origin plus source-local timestamp, retained unchanged',
                'no_near_time_merge':True,'no_semantic_mapping':True,'all_original_text_retained':True}}
    source_payload(record)
    if validate_plan:plan_schema(record)
    return record

def main():
    jobs=[];mapping=[]
    for line in SOURCE.read_text().splitlines():
        bundle=json.loads(line);record=prepare(bundle);job={'job_id':bundle['sample_id'],'record':record,
            'provenance':{'role':'native8_nod_visual_pilot','source_bundle_sha256':digest(bundle),
                          'source_file':str(SOURCE),'media_reviewed_by_this_stage':False}}
        jobs.append(job);save(OUT/'inputs'/f"{bundle['sample_id']}.json",record)
        mapping.append({'sample_id':bundle['sample_id'],'original':record['preprocessing']['original_observations'],
                        'light_dedup':len(record['events']),'exact_duplicates':record['preprocessing']['exact_duplicates_collapsed'],
                        'nod_candidates':len(bundle['nod_candidates']),'invalid_windows':len(record['preprocessing']['invalid_responses'])})
    with (OUT/'jobs.jsonl').open('w') as f:
        for job in jobs:f.write(json.dumps(job,ensure_ascii=False)+'\n')
    save(OUT/'preprocessing.json',{'samples':mapping,'source_sha256':sha(SOURCE),
        'plan_prompt':PLAN_PROMPT,'review_prompt':REVIEW_PROMPT,'preparer_sha256':sha(Path(__file__)),
        'raw_count':sum(r['original'] for r in mapping),'light_count':sum(r['light_dedup'] for r in mapping)})
    print(json.dumps({'samples':len(jobs),'raw':sum(r['original'] for r in mapping),'after_exact_dedup':sum(r['light_dedup'] for r in mapping)}))

if __name__=='__main__':main()
