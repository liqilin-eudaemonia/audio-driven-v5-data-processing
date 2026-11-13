#!/usr/bin/env python3
"""Read only benchmark result files; compare input identity and raw outputs."""
import argparse
import json
from pathlib import Path
import statistics


def read_arm(path):
    rows=[json.loads(p.read_text()) for p in sorted((path/'results').glob('*.json'))]
    return rows,{(r['sample_id'],v['view']):v for r in rows for v in r['records']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);args=p.parse_args()
    reference,ref=read_arm(args.reference)
    summary=json.loads((args.run/'summary.json').read_text());report={}
    for name,timing in summary.items():
        rows,records=read_arm(args.run/name)
        shared=records.keys()&ref.keys();same_input=0;same_text=0;changed=[]
        for key in sorted(shared):
            a,b=ref[key],records[key]
            equal=all(a[k]==b[k] for k in ('start','end','prompt','rendered_prompt','media','native','prompt_tokens'))
            same_input+=equal;same_text+=a['raw']==b['raw']
            if a['raw']!=b['raw']:
                changed.append({'sample_id':key[0],'view':key[1],
                    'reference_status':a['status'],'status':b['status'],
                    'reference_events':len(a.get('parsed',{}).get('events',[])),
                    'events':len(b.get('parsed',{}).get('events',[]))})
        totals={field:sum(r[field] for r in timing['samples_detail']) for field in
                ('requests','valid','events','output_tokens','prompt_tokens','generate_s','decode_s','prepare_s')}
        # Same evidence and request IDs are required even for invalid outputs.
        report[name]={**{k:v for k,v in timing.items() if k!='samples_detail'},**totals,
            'all_reference_requests_present':records.keys()==ref.keys(),
            'identical_input_requests':same_input,'identical_raw_responses':same_text,
            'reference_requests':len(ref),'changes':changed}
    resources=[json.loads(line) for line in (args.run/'resources.jsonl').read_text().splitlines()]
    for name,r in report.items():
        gpu=r['gpu'];active=[point['gpus'][gpu] for point in resources if point['gpus'][gpu]['utilization']>0]
        r['gpu_utilization_mean_when_nonzero']=statistics.mean(g['utilization'] for g in active) if active else 0
        r['peak_memory_mib']=max((p['gpus'][gpu]['memory_used_mib'] for p in resources),default=0)
    (args.run/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name,r in sorted(report.items(),key=lambda kv:-(kv[1]['clips_per_s'] or 0)):
        print(name,{k:r[k] for k in ('samples','active_wall_s','clips_per_s','invalid_requests','identical_input_requests',
                                    'identical_raw_responses','events','peak_memory_mib','exit_code')})


if __name__=='__main__':main()
