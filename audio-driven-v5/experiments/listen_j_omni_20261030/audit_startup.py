from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni import read_audio,file_sha
from listen_benchmark.native_caption_nod import render
from listen_benchmark.native_caption_quality import generation_failures
w=Path(__file__).resolve().parent;cfg=read(w/'config.json');q=read(w/'queues/caption/state.json')
jobs=set(q['inflight'])
with (w/'ready.jsonl').open() as f:
 for _ in range(200):
  if f.tell()>=q['offset']:break
  line=f.readline()
  if not line:break
  jobs.add(json.loads(line)['job_id'])
selected={}
for jid in jobs:
 job=read(w/'jobs'/f'{jid}.json')
 for row in job['rows']:
  p=result_path(w,'caption',row['sample_id'])
  if not p.exists():continue
  r=read(p);key=(r['node'],r['gpu'])
  if r['node'] not in cfg['nodes'] or key in selected:continue
  selected[key]=(p,r)
 if len(selected)==16:break
assert len(selected)==16,sorted(selected)
checks=[]
for (node,gpu),(p,r) in sorted(selected.items()):
 ev=r['input_evidence'];wave,sr,identity=read_audio(r['audio_recipe']);assert identity==ev['audio_identity']
 assert r['pipeline_id']==cfg['pipeline_id'] and r['generation_checks_passed'] and not generation_failures(r['caption'],r['finish_reason'])
 row=dict(sample_id=r['sample_id'],duration_s=r['source']['duration_s'],nod_supplement=ev['nod_supplement'],audio_preflight={'full_silence':identity['full_silence']})
 assert render(row,style='first_plus_evidence')==r['prompt']
 nod=read(ev['nod_supplement']['detector_file']);assert nod['source']==r['source'] and file_sha(ev['nod_supplement']['detector_file'])==ev['nod_supplement']['detector_sha256']
 assert r['audio_recipe']['source_time_origin_s']==ev['video_evidence']['source_time_origin_s']
 checks.append(dict(node=node,gpu=gpu,sample_id=r['sample_id'],file=str(p),sha256=file_sha(p),full_silence=identity['full_silence'],caption_words=len(r['caption'].split()),verified=True))
report=dict(passed=True,at=time.time(),sampled_outputs=16,all16_gpus_verified=True,checks=checks,semantic_quality_verified=False)
write(w/'validation/production_startup_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
