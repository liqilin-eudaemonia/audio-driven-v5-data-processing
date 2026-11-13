"""Read fixed workers and their current sources, never recursively scan results."""
import hashlib,json,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[3];w=r/'experiments/listen_j_omni_20261030';sys.path.insert(0,str(r/'src'))
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_visual_gate import QualityGate
cfg=read(w/'config.json');gate=QualityGate(cfg['visual_quality_gate']);out=dict(at=time.time(),workers=[],failures=[])
for n,nc in cfg['nodes'].items():
 for gpu in nc['gpu_ids']:
  h=read(w/'workers'/f'{n}-gpu-{gpu}.json');record=dict(node=n,gpu=gpu,pid=h['pid'],stage=h['stage'],completed=h['completed_clips'],revision=h.get('execution_revision'),heartbeat_age=time.time()-h['heartbeat'])
  assert 'visual-gate-v1' in h['execution_revision'] and record['heartbeat_age']<60
  assert h['failed_clips']==0 and not h.get('last_error'),h
  if not h.get('job'):out['failures'].append([n,gpu,'no source yet']);continue
  job=read(w/'jobs'/f"{h['job']}.json");decisions=gate.decisions(job);counts=dict(caption=0,excluded=0,review=0,legacy=0)
  for row in job['rows']:
   sid=row['sample_id'];d=decisions[sid];p=result_path(w,'caption',sid)
   if p.exists():
    result=read(p)
    if result.get('visual_quality') is not None:
     assert result['visual_quality']==d and d['decision']=='retain'
     assert hashlib.sha256(Path(d['measurement_file']).read_bytes()).hexdigest()==d['measurement_sha256']
     counts['caption']+=1
    else:counts['legacy']+=1
   for b,key,decision in [('visual_excluded','excluded','exclude'),('visual_review','review','review')]:
    p=result_path(w,b,sid)
    if p.exists():
     assert d['decision']==decision and read(p)['visual_quality']==d
     counts[key]+=1
  record['current_source']=h['job'];record['counts']=counts;out['workers'].append(record)
out['passed']=len(out['workers'])==sum(len(c['gpu_ids']) for c in cfg['nodes'].values()) and not out['failures']
write(w/'visual_quality_rollout/cluster_gate_audit.json',out)
print(json.dumps(dict(passed=out['passed'],workers=len(out['workers']),counts={k:sum(x['counts'][k] for x in out['workers']) for k in ['caption','excluded','review','legacy']})))
