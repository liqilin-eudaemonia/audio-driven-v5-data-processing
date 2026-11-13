"""Bounded live audit: current per-card sources, no recursive result scan."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni import EXECUTION_REVISION,read_audio,file_sha
from listen_benchmark.native_caption_nod import render
from listen_benchmark.native_caption_quality import generation_failures
cfg=read(WORK/'config.json');expected={(n,g) for n,c in cfg['nodes'].items() for g in c['gpu_ids']};jobs=set();workers=[]
for n,g in sorted(expected):
 h=read(WORK/'workers'/f'{n}-gpu-{g}.json');assert h.get('execution_revision')==EXECUTION_REVISION,(n,g,h['stage']);assert time.time()-h['heartbeat']<45
 assert h['completed_clips']>0,(n,g,'no output yet');assert not h.get('last_error'),(n,g,h['last_error'][-500:])
 if h.get('job'):jobs.add(h['job'])
 workers.append(dict(node=n,gpu=g,stage=h['stage'],completed=h['completed_clips']))
selected={}
selection_file=WORK/'validation/v2_audit_selected.json'
if selection_file.exists():
 for name in read(selection_file):
  p=Path(name);r=read(p);key=(r['node'],r['gpu'])
  if key in expected and r.get('execution_revision')==EXECUTION_REVISION:selected[key]=(p,r)
for jid in sorted(jobs):
 for row in read(WORK/'jobs'/f'{jid}.json')['rows']:
  p=result_path(WORK,'caption',row['sample_id'])
  if not p.exists():continue
  r=read(p);key=(r['node'],r['gpu'])
  if key not in expected or key in selected or r.get('execution_revision')!=EXECUTION_REVISION:continue
  selected[key]=(p,r)
write(selection_file,[str(p) for p,r in selected.values()])
assert set(selected)==expected,dict(missing=sorted(expected-set(selected)))
checks=[]
for (n,g),(p,r) in sorted(selected.items()):
 ev=r['input_evidence'];wave,sr,identity=read_audio(r['audio_recipe']);assert identity==ev['audio_identity']
 assert r['pipeline_id']==cfg['pipeline_id'] and r['generation_checks_passed'] and not generation_failures(r['caption'],r['finish_reason'])
 row=dict(sample_id=r['sample_id'],duration_s=r['source']['duration_s'],nod_supplement=ev['nod_supplement'],audio_preflight={'full_silence':identity['full_silence']})
 assert render(row,style='first_plus_evidence')==r['prompt']
 nod=read(ev['nod_supplement']['detector_file']);assert nod['source']==r['source'] and file_sha(ev['nod_supplement']['detector_file'])==ev['nod_supplement']['detector_sha256']
 assert r['audio_recipe']['source_time_origin_s']==ev['video_evidence']['source_time_origin_s']
 checks.append(dict(node=n,gpu=g,sample_id=r['sample_id'],file=str(p),sha256=file_sha(p),silent=identity['full_silence'],verified=True))
q=read(WORK/'queues/caption/state.json');assert not q['failed'],list(q['failed'])
archives=sorted((WORK/'revisions'/EXECUTION_REVISION).glob('failed-before-requeue-*.json'));assert archives
recovered={}
for p in archives:
 for jid in read(p):
  receipt=WORK/'caption/completed'/f'{jid}.json';assert receipt.exists(),('not recovered',jid)
  r=read(receipt);recovered[jid]=dict(clips=r['clips'],held=r['held_clips'],failed_generation=r['failed_generation_clips'])
assert read(WORK/'validation/node4_lock.json')['passed']
report=dict(passed=True,at=time.time(),execution_revision=EXECUTION_REVISION,gpu_count=len(expected),workers=workers,checks=checks,recovered_sources=recovered,resolved_clips=q['completed_clips'],failed_sources=0,semantic_quality_verified=False)
write(WORK/'validation/production_v2_24gpu_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['workers','checks','recovered_sources']}));print('recovered sources',len(recovered))
