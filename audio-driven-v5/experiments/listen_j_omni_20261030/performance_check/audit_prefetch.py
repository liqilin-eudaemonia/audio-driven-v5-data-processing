"""Audit current fast outputs without scanning result trees."""
import sys,time,hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path,adapt_clip
from listen_benchmark.j_omni import read_audio,file_sha,EXECUTION_REVISION
expected_revision=EXECUTION_REVISION+'+process-prefetch-v1'
cfg=read(W/'config.json');records={};path=W/'performance_check/prefetch_audit_selected.json'
if path.exists():records=read(path)
records={k:v for k,v in records.items() if v.get('execution_revision')==expected_revision}
ready=[];not_ready=[]
for node,nc in cfg['nodes'].items():
 for gpu in nc['gpu_ids']:
  key=f'{node}-gpu-{gpu}';profile=cfg.get('preparation_profiles',{}).get(key);hp=W/'workers'/f'{key}.json';h=read(hp)
  if not profile or h.get('preparation_profile')!=profile or h.get('execution_revision')!=expected_revision or h['completed_clips']<5 or time.time()-h['heartbeat']>45 or h.get('last_error'):
   not_ready.append(dict(worker=key,stage=h['stage'],completed=h['completed_clips'],error=(h.get('last_error') or '')[-200:]));continue
  ready.append(key)
  existing=records.get(key)
  if existing:
   r=read(existing['file'])
   if r.get('preparation_profile')==profile and r.get('execution_revision')==expected_revision:continue
   # An older configuration's evidence cannot satisfy the new rollout audit.
   records.pop(key,None)
  job=read(W/'jobs'/f"{h['job']}.json")
  for row in job['rows']:
   p=result_path(W,'caption',row['sample_id'])
   if not p.exists():continue
   r=read(p)
   if r.get('node')!=node or r.get('gpu')!=gpu or r.get('preparation_profile')!=profile or r.get('execution_revision')!=expected_revision:continue
   s=r['source'];original=read(s['source_result']);receipt=read(Path(s['source_result']).parent/'complete.json');assert file_sha(s['source_result'])==s['source_result_sha256']==receipt['result_sha256']
   c=original['clips'][s['source_clip_index']];rebuilt=adapt_clip(c,{k:s.get(k) for k in ['pair','split','source_catalog_status']},s['source_result'],s['source_result_sha256'],s['source_clip_index']);assert rebuilt['source']==s
   wave,sr,identity=read_audio(r['audio_recipe']);assert identity==r['input_evidence']['audio_identity']
   assert r['preparation_metrics']['cuda_visible_devices']=='' and r['preparation_metrics']['pid']!=h['pid']
   assert r['input_evidence']['video_evidence']['video_transform']==s['video_transform']
   assert r['pipeline_id']==cfg['pipeline_id'] and r['generation_checks_passed']
   records[key]=dict(file=str(p),sample_id=r['sample_id'],profile=profile,execution_revision=expected_revision,source_and_audio_verified=True,cpu_child_cuda_hidden=True);break
write(path,records);q=read(W/'queues/caption/state.json')
r=dict(at=time.time(),expected_revision=expected_revision,ready=ready,not_ready=not_ready,verified_cards=len(records),missing_verified_cards=sorted(set(ready)-set(records)),failed_sources=len(q['failed']),resolved_clips=q['completed_clips'],all_code_hashes_match=all(file_sha(ROOT/p)==h for p,h in cfg['code_hashes'].items()))
r['passed']=len(ready)==24 and len(records)==24 and not q['failed'] and r['all_code_hashes_match']
write(W/'validation/prefetch_production_audit.json',r);print(json.dumps(r,ensure_ascii=False))
