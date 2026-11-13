"""Compare GPU7 work/input costs in two already completed production windows."""
import json,statistics,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent

def describe(values):
 return {'n':len(values),'median':statistics.median(values),'mean':statistics.mean(values),'max':max(values)} if values else None

def extract(tag):
 report=json.loads((W/f'monitor_snapshots/{tag}.json').read_text());assert report['complete'] and report['summary']['valid_for_throughput_comparison']
 first,last=report['samples'][0],report['samples'][-1];q0,q1=first['vlm_queue_range'],last['vlm_queue_range'];assert q1['offset']>=q0['offset']
 jobs=set(q0['inflight'])
 with (W/'ready.jsonl').open('rb') as f:
  f.seek(q0['offset']);jobs.update(json.loads(line)['job_id'] for line in f.read(q1['offset']-q0['offset']).splitlines())
 metrics={};groups={};sizes={};seconds=[];seen=set();cache_hits=0;outputs=prompts=requests=0
 for j in sorted(jobs):
  for row in json.loads((W/f'jobs/{j}.json').read_text())['rows']:
   sid=row['sample_id']
   if sid in seen:continue
   seen.add(sid);p=W/'vlm/results'/hashlib.sha256(sid.encode()).hexdigest()[:2]/f'{sid}.json'
   if not p.exists():continue
   d=json.loads(p.read_text())
   if not first['at']<=d['completed_at']<last['at'] or d['execution'].get('worker_id')!='gpu-7':continue
   assert d['source']==row['source'];seconds.append(d['duration_s']);size=str(d['source']['video_transform']['source_size']);sizes[size]=sizes.get(size,0)+1
   perf=d['input_performance'];cache_hits+=bool(perf.get('media_cache_hit'))
   for k,v in perf.items():
    if isinstance(v,(float,int)) and not isinstance(v,bool) and (not perf.get('media_cache_hit') or k not in ['decode_s','decode_lock_wait_s']):metrics.setdefault(k,[]).append(v)
   g=d['prefetch_group_performance'];groups[json.dumps(g,sort_keys=True)]=g
   for r in d['records']:requests+=1;prompts+=r['prompt_tokens'];outputs+=r['completion_tokens']
 return {'tag':tag,'clips':len(seconds),'seconds':sum(seconds),'mean_seconds':statistics.mean(seconds),'sizes':sizes,'cache_hits':cache_hits,'requests':requests,'prompt_tokens':prompts,'output_tokens':outputs,'metrics':{k:describe(v) for k,v in metrics.items() if k!='decoder_pid'},'group_prepare_s':describe([g['total_prepare_wall_s'] for g in groups.values() if 'total_prepare_wall_s' in g]),'summary':report['summary']}

if __name__=='__main__':
 result={'thread':extract('decode_graph_stable_window'),'process':extract('process_decode_stable_window'),'scope':'Different production inputs; compare workload and preparation costs, not a matched end-to-end speedup.'}
 (W/'monitor_snapshots/decode_backend_production_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
