import collections,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent

def support(parsed):
 result={}
 for key,r in parsed['reviewed'].items():
  u=parsed['unknown'][key]+[[e['start'],e['end']] for e in parsed['events'] if e.get('type')==key and e.get('boundary_censored')]
  points=sorted(set(t for span in r+u for t in span));result[key]=sum(b-a for a,b in zip(points,points[1:]) if any(x<=a and b<=y for x,y in r) and not any(x<b and a<y for x,y in u))
 return result
runs={}
for run in ['comparison50','comparison_guided12','comparison_best50']:
 p=root/run/'summary.json';s=json.loads(p.read_text());systems={}
 for name in sorted({r['system'] for r in s['results']}):
  rows=[r for r in s['results'] if r['system']==name];good=[r for r in rows if r['status']=='schema_pass'];by_sample={}
  for r in good:by_sample.setdefault(r['sample_id'],{})[r['stage']]=r
  complete={sid:stages for sid,stages in by_sample.items() if len(stages)==2}
  known={sid:stages for sid,stages in complete.items() if all(v>0 for v in support(stages['audit']['parsed']).values())}
  durations=[]
  for f in (root/run/'requests'/name).glob('**/attempt-*.json'):
   try:durations.append((f.stat().st_mtime_ns-int(f.stem.split('-')[1]))/1e9)
   except Exception:pass
  audits=[r for r in good if r['stage']=='audit']
  systems[name]=dict(requests=len(rows),schema_pass_requests=len(good),complete_samples=len(complete),
   complete_with_nonzero_audit_support=len(known),complete_sample_ids=sorted(complete),supported_sample_ids=sorted(known),
   claimed_audit_events=sum(len(r['parsed']['events']) for r in audits),
   inspected_minus_unknown_seconds={k:sum(support(r['parsed']).get(k,0) for r in audits) for k in {k for r in audits for k in r['parsed']['reviewed']}},
   failure_reasons=dict(collections.Counter(r['error'] for r in rows if r['status']!='schema_pass')),
   observed_http_seconds_median=statistics.median(durations) if durations else None,
   observed_http_seconds_max=max(durations) if durations else None)
 runs[run]=dict(complete=s['complete'],completed_requests=s['completed'],systems=systems)
result=dict(ground_truth_available=False,accuracy_or_recall_measured=False,selection='Joy visual + Qwen3-Omni audio-only + constrained JSON + independent discovery/audit',
 caveat='Support and events are model claims, not verified truth. HTTP time includes server queuing on shared GPUs; not a fair isolated speed benchmark. Semantic reinspection retries were added to production after these comparisons.',runs=runs)
(root/'model_comparison_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for run,r in runs.items():
 print(run,r['complete'])
 for name,v in r['systems'].items():print(name,v['schema_pass_requests'],v['complete_samples'],v['complete_with_nonzero_audit_support'],v['claimed_audit_events'])
