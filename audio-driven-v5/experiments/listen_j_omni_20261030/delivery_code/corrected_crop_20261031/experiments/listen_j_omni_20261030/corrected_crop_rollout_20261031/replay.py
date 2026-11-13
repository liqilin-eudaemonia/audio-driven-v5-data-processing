import os
os.environ['J_OMNI_BOUNDED_CROP_V1']='1'
import sys,json,time,hashlib,concurrent.futures
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni import prepare_input,file_sha
from listen_benchmark.j_omni_crop import effective_row
W=R/'experiments/listen_j_omni_20261030';D=W/'corrected_crop_rollout_20261031';C=read(D/'candidate_config.json')
write(D/'config.json',C)
items=read(W/'input_chain_audit_20261031/selected.json')['samples']
def check(item):
 old=read(item['path']);job=read(W/'jobs'/f"{item['job_id']}.json");row=next(r for r in job['rows'] if r['sample_id']==item['sample_id'])
 origin=read(W/'job_origins'/f"{item['job_id']}.json")
 p=prepare_input(row,origin['old_work'],D,C['nodes']['node12']['resource_work'])
 eff,prov=effective_row(row,C);nod=read(p['evidence']['nod_supplement']['detector_file'])
 assert p['effective_source']==eff['source']
 assert nod['media']['video_transform']==eff['source']['video_transform']
 assert p['evidence']['video_evidence']['video_transform']==eff['source']['video_transform']
 assert p['recipe']==old['audio_recipe']
 for key in ['audio_encoder_sha256','audio_encoder_samples','audio_identity']:
  assert p['evidence'][key]==old['input_evidence'][key],key
 assert p['frames'].shape[1:3]==(512,512)
 assert file_sha(item['path'])==item['result_sha256'],'Old label was changed'
 out=dict(sample_id=item['sample_id'],crop=prov,nod_source_matches_actual=nod['source']==eff['source'] if prov['coordinates_changed'] else nod['source']==row['source'],nod_reuse=p['evidence']['nod_supplement']['crop_reuse'],audio_unchanged=True,original_caption_unchanged=True,frames=len(p['frames']),passed=True)
 write(D/'replay'/f"{item['sample_id']}.json",out);return out
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(check,i) for i in items]
 for f in futures:
  results.append(f.result());print(results[-1]['sample_id'],flush=True)
write(D/'replay.json',dict(at=time.time(),passed=True,samples=results));print('ALL PASSED',flush=True)
