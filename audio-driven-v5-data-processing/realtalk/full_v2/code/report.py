from pathlib import Path
import json,time,collections,sys,html
R=Path(__file__).resolve().parents[1];O=R.parent/'full_20261023/reports/full_v2';O.mkdir(parents=True,exist_ok=True)
from pipeline import atomic
states=[]
for p in (R/'state').glob('*.json'):
 try:states.append(json.loads(p.read_text()))
 except Exception:pass
counts=collections.Counter(s['stage'] for s in states);summary={'updated_at':time.time(),'source_count':692,'source_stages':dict(counts),'source_states':states,'all_sources_complete':len(states)==692 and counts['complete']==692,'candidate_pairs':0,'candidate_seconds':0,'classified_seconds':{},'max_pair_seconds':0,'scope':'new v2 manifests; original 129094 pairs not used as work queue','resolution_threshold':None,'syncnet_pass_is_not_clean_audio_or_semantic_certification':True}
for p in (R/'manifests').glob('*.json'):
 m=json.loads(p.read_text());s=m['summary'];summary['candidate_pairs']+=s['pair_count'];summary['candidate_seconds']+=s['candidate_seconds'];summary['max_pair_seconds']=max(summary['max_pair_seconds'],s['max_pair_seconds'])
 for k,v in s['ledger_seconds'].items():summary['classified_seconds'][k]=summary['classified_seconds'].get(k,0)+v
# Expensive result aggregation runs at most every ten minutes, or at finalization.
cache=R/'reports/score_counts.json'
if '--final' in sys.argv or not cache.exists() or time.time()-cache.stat().st_mtime>600:
 cats=collections.Counter();duration=collections.Counter();passes={'raw':[],'optimized':[]};records=0
 for f in (R/'results').glob('*/*.json'):
  try:r=json.loads(f.read_text())
  except Exception:continue
  records+=1
  for mode in ['raw','optimized']:
   s=r.get(mode+'_sync',{});label=mode+':'+s.get('status',r.get('status',r.get('phase','unknown')));cats[label]+=1;duration[label]+=r['duration']
   if s.get('robust_pass'):
    cats[mode+':syncnet_pass']+=1;duration[mode+':syncnet_pass']+=r['duration'];passes[mode].append({'pair_id':r['pair_id'],'video_id':r['video_id'],'duration':r['duration'],'role_review_required':r.get('role_review_required',True),'result_file':str(f)})
  if r.get('role_review_required',True):cats['role_or_identity_review']+=1
 atomic(cache,{'updated_at':time.time(),'records':records,'counts':dict(cats),'seconds':dict(duration)})
 for mode,rows in passes.items():atomic(R/'reports'/f'{mode}_syncnet_pass.json',rows)
summary['scores']=json.loads(cache.read_text());atomic(R/'reports/progress.json',summary);atomic(O/'progress.json',summary)
links={'manifests':R/'manifests','raw_syncnet_pass.json':R/'reports/raw_syncnet_pass.json','optimized_syncnet_pass.json':R/'reports/optimized_syncnet_pass.json'}
for name,target in links.items():
 p=O/name
 if not p.exists() and not p.is_symlink():p.symlink_to(target)
page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><style>body{font:17px sans-serif;max-width:1100px;margin:40px auto;padding:20px;background:#14232d;color:#eee}a{color:#8ee4ef}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>RealTalk 全量 v2</h1><p>新切分 → listener 画面和本人分离音轨 → 优化前后各自 SyncNet。短回应保留、正式接话切开并交换角色继续；未知区间单列。未应用分辨率阈值。</p><p>分数通过不代表人工确认没有串音；短片段、静音、身份待确认、技术异常分别记录。</p><a href="samples.html">抽样前后对照（生成后可用）</a> · <a href="progress.json">完整进度 JSON</a><pre>'''+html.escape(json.dumps({k:v for k,v in summary.items() if k!='source_states'},ensure_ascii=False,indent=2))+'</pre>'
(O/'index.html').write_text(page)
