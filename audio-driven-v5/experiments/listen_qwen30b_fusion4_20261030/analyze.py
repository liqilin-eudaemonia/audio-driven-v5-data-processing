"""Paired text/evidence analysis; no claims of video-grounded accuracy."""
from collections import Counter
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def output_key(result):
    return sorted((round(e['start'],9),round(e['end'],9),e['description'])
                  for e in result['events'] if e['modality']=='visual')


def analyze_arm(arm):
    jobs=[json.loads(l) for l in (WORK/f'{arm}.jsonl').read_text().splitlines()]
    run=WORK/'runs'/arm;summary=read(run/'summary.json');audit=read(run/'audit.json')
    accepted={};results={};tokens=0;truncations=0;requests=0;nod_inputs=nod_represented=0
    samples=[]
    for j in jobs:
        key=j['job_id'];p=run/'results'/f'{key}.json'
        if p.exists():
            result=read(p);results[key]=result;accepted[key]=j
            decisions={d['source_event_id']:d for d in result['source_decisions']}
            for e in j['record']['events']:
                if e.get('source_type')!='glare_nod_recall':continue
                count=len(e.get('original_observations',[e]));nod_inputs+=count
                d=decisions[e['event_id']]
                if d['disposition'] in ('represented','redundant') and d['target_event_ids']:nod_represented+=count
            samples.append({'sample_id':key,'input_count':len(j['record']['events']),
                            'output_count':len(result['events']),'output':output_key(result)})
        for attempt in range(3):
            for stage in ['proposal','review']:
                p=run/'attempts'/f'{key}.{stage}.{attempt}.json'
                if p.exists():
                    a=read(p);requests+=1;tokens+=a['tokens'];truncations+=a['finish_reason']!='stop'
    if nod_inputs!=nod_represented:raise ValueError(f'Nod evidence lost: {arm}')
    if audit['verified']!=summary['accepted']:raise ValueError('Accepted/replay count differs')
    row={'samples':len(jobs),'input_events_all':sum(len(j['record']['events']) for j in jobs),
         'accepted':summary['accepted'],'rejected':len(summary['rejected']),
         'rejection_stages':dict(Counter(x['stage'] for x in summary['rejected'])),
         'input_events_accepted':sum(len(j['record']['events']) for j in accepted.values()),
         'output_events_accepted':sum(len(x['events']) for x in results.values()),
         'cpu_replay_verified':audit['verified'],'nod_observations_in_accepted_inputs':nod_inputs,
         'nod_observations_represented':nod_represented,'requests':requests,'generated_tokens':tokens,
         'truncated_responses':truncations,'elapsed_s_including_model_startup':summary['elapsed_seconds'],
         'rejections':summary['rejected'],'samples_detail':samples}
    return row,results


def main():
    metrics={};outputs={}
    for arm in ['prior15_direct','prior15_light','j50_direct','j50_light']:
        metrics[arm],outputs[arm]=analyze_arm(arm)
    paired={}
    for group in ['prior15','j50']:
        direct=outputs[group+'_direct'];light=outputs[group+'_light'];common=direct.keys()&light.keys()
        identical=[sid for sid in sorted(common) if output_key(direct[sid])==output_key(light[sid])]
        paired[group]={'both_accepted':len(common),'identical_outputs':len(identical),
            'direct_only_accepted':sorted(direct.keys()-light.keys()),'light_only_accepted':sorted(light.keys()-direct.keys()),
            'different_both_accepted':[sid for sid in sorted(common) if sid not in identical],
            'both_accepted_direct_events':sum(len(direct[s]['events']) for s in common),
            'both_accepted_light_events':sum(len(light[s]['events']) for s in common)}
    report={'arms':metrics,'paired':paired,'scope':'Text-only program/model review and provenance replay. Single simultaneous run per arm; no semantic accuracy or causal speedup claim.'}
    (WORK/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'arms':{k:{n:v for n,v in x.items() if n not in ('rejections','samples_detail')} for k,x in metrics.items()},'paired':paired},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
