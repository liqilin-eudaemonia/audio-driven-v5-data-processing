"""Bounded, single-process diagnostic of recent node8 workload; never a tree scan."""
import json,sys,statistics,time
from pathlib import Path
w=Path(__file__).resolve().parents[1];root=w.parents[1];sys.path.insert(0,str(root/'src'))
from listen_benchmark.j_annotation import read,write,result_path
folders=['node8_released_cpu_candidate32'];windows={f:read(w/'visual_quality_rollout'/f/'node8.json') for f in folders if (w/'visual_quality_rollout'/f/'node8.json').exists()}
lo=min(x['start'] for x in windows.values());hi=max(x['end'] for x in windows.values());q=read(w/'queues/caption/state.json')
with (w/'ready.jsonl').open('rb') as f:
 start=max(0,q['offset']-150000);f.seek(start);raw=f.read(q['offset']-start).splitlines();raw=raw[1:] if start else raw
ids=list(dict.fromkeys([json.loads(x)['job_id'] for x in raw][-1000:]+list(q['inflight'])))
rows={f:[] for f in windows};reads=0;sources=0
for jid in ids:
 receipt=w/'caption/completed'/f'{jid}.json'
 if receipt.exists() and not lo<=receipt.stat().st_mtime<=hi+240:continue
 job=read(w/'jobs'/f'{jid}.json');available=[]
 for row in job['rows']:
  p=result_path(w,'caption',row['sample_id'])
  if p.exists():available.append(p)
 if not available:continue
 head=read(available[0]);reads+=1
 if head.get('node')!='node8':continue
 sources+=1
 for p in available:
  d=head if p==available[0] else read(p);reads+=p!=available[0]
  if reads>3000:break
  if d.get('node')!='node8':continue
  for f,x in windows.items():
   if x['start']<=d['completed_at']<=x['end']:
    rows[f].append({'sample_id':d['sample_id'],'duration':d['source']['duration_s'],'frames':len(d['input_evidence']['frame_pts_s']),'prompt_tokens':d['attempts'][-1]['prompt_tokens'],'completion_tokens':d['completion_tokens'],'preparation_s':d['preparation_metrics']['seconds']})
 if reads>3000:break
summary={}
for f,rs in rows.items():
 x=windows[f];summary[f]={'captions_observed':x['clips'],'captions_sampled':len(rs),'coverage':len(rs)/x['clips'],'means':{k:statistics.mean(r[k] for r in rs) for k in ['duration','frames','prompt_tokens','completion_tokens','preparation_s']} if rs else {}}
out={'at':time.time(),'source_ids_checked':len(ids),'node8_sources_read':sources,'caption_files_read':reads,'summary':summary,'scope':'Recent journal tail/current sources only; partial coverage is explicit; no causal attribution from unequal clips.'}
write(w/'visual_quality_rollout/node8_released_cpu32_workload.json',out);print(json.dumps(out,indent=2))
