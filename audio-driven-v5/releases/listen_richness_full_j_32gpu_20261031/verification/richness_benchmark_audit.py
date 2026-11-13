import sys,json,hashlib,time,collections,statistics
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(r/'src'),str(r/'scripts')]
from listen_richness.production import statistic_record
from listen_richness.common import digest
w=r/'experiments/listen_richness_full_j_20261031';result=w/'benchmark_32_result.json'
if not result.exists():raise SystemExit('benchmark not yet complete')
r0=json.loads(result.read_text());counts=collections.Counter();seen=set();checked=0;sample_per_gpu={}
for p in (w/'extract/completed').glob('*.json'):
 rec=json.loads(p.read_text());counts.update({k:rec[k] for k in ['clips','strict_failed','incomplete_generations','unresolved']})
 for key in ('raw','statistics'):
  assert hashlib.sha256(Path(rec[key+'_file']).read_bytes()).hexdigest()==rec[key+'_sha256']
 gpu=(rec['node'],rec['gpu'])
 if gpu in sample_per_gpu:continue
 raw=json.loads(Path(rec['raw_file']).open().readline());stat=json.loads(Path(rec['statistics_file']).open().readline());job=json.loads((w/'jobs'/p.name).read_text());request=json.loads(Path(job['path']).open().readline())
 assert raw['sample_id']==stat['sample_id']==request['sample_id'];assert stat['caption']==request['snapshot']['caption'];assert stat['source']==request['snapshot']['source']
 again=statistic_record(request,raw['attempts'],rec['raw_file'])
 for key in ['counted_types','behavior_quantities','known_occurrences_by_type','evidence_ledger','record_status']:assert again[key]==stat[key]
 sample_per_gpu[gpu]=stat['sample_id'];checked+=1
assert checked==32 and counts['clips']==r0['completed'] and counts['unresolved']==0 and counts['incomplete_generations']==0
resources={}
for n in ['node17','node8','node4','node12']:
 rr=[json.loads(l) for l in (w/'resources'/f'{n}.jsonl').read_text().splitlines()]
 active=[x for x in rr if r0['started_at']<=x['at']<=r0['started_at']+180]
 resources[n]=dict(gpu_mean=statistics.mean(g[1] for x in active for g in x['gpus']),memory_mib_mean=statistics.mean(g[2] for x in active for g in x['gpus']),cpu_cores_mean=statistics.mean(x['cpu_cores'] for x in active))
audit=dict(at=time.time(),passed=True,counts=dict(counts),source_replays=checked,source_samples={f'{n}-gpu-{g}':s for (n,g),s in sample_per_gpu.items()},active_window_resources=resources,output_hashes_verified=True)
(w/'benchmark_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(audit,ensure_ascii=False))
