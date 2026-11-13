import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,hashlib,sys,tarfile,time
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path,adapt_clip
from listen_benchmark.j_omni import prepare_input,read_audio,fingerprint
from listen_benchmark.official_audio_gate import make_gate
from listen_audio_postprocess.bulk import official
from listen_benchmark.native_caption_nod import render
from listen_benchmark.j_omni_queued import request
from transformers import AutoProcessor
processor=AutoProcessor.from_pretrained('/private/ckpts/Qwen3-Omni/Instruct',trust_remote_code=True)
node=sys.argv[1];samples=[x for x in read(W/'input_chain_audit_20261031/selected.json')['samples'] if x['node']==node];cfg=read(W/'config.json');out=[]
for sample in samples:
 tick=time.time();result=read(sample['path']);assert hashlib.sha256(Path(sample['path']).read_bytes()).hexdigest()==sample['result_sha256']
 sid=sample['sample_id'];source=result['source'];job=read(W/'jobs'/f"{source['pair']}.json");row=next(x for x in job['rows'] if x['sample_id']==sid)
 assert source==row['source'] and result['pipeline_id']==cfg['pipeline_id']
 srcpath=Path(source['source_result']);assert srcpath.is_relative_to(R/'data_corpora/seamless_listen_j_full_20261029/results')
 raw=srcpath.read_bytes();assert hashlib.sha256(raw).hexdigest()==source['source_result_sha256']==job['source_result_sha256']
 origin=json.loads(raw);clip=origin['clips'][source['source_clip_index']];assert clip['sample_id']==sid
 receipt=read(srcpath.parent/'complete.json');assert receipt['result_sha256']==source['source_result_sha256'] and origin['complete']
 rebuilt=adapt_clip(clip,dict(pair=source['pair'],split=source['split'],source_catalog_status=source['source_catalog_status']),srcpath,source['source_result_sha256'],source['source_clip_index']);assert rebuilt==row
 assert clip['sources']['listener']!=clip['sources']['speaker'] and source['listener_id']!=source['speaker_id']
 assert clip['normalized_audio']['listener']['contract']['source']==clip['sources']['listener']
 assert source['video_transform']['output_size']==[512,512]
 recipe=result['audio_recipe'];meta_path=str(Path(source['source']['member']).with_suffix('.json'))
 with tarfile.open(source['source']['archive'],'r:') as archive:
  meta=archive.extractfile(meta_path).read()
 assert hashlib.sha256(meta).hexdigest()==recipe['provenance']['sha256'] and recipe['provenance']['member']==meta_path
 start=recipe['source']['first_sample']/recipe['source']['sample_rate'];duration=source['duration_s'];off=official(json.loads(meta),start,duration)
 normalized={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in off[k]] for k in ['words','sentences','vad']}
 assert normalized==recipe['official_listener'];assert recipe['language_filter']=='disabled_user_official_union_only' and recipe['extra_mute_intervals']==[]
 processed,sr,audio_id=read_audio(recipe);mask,gate=make_gate(normalized,len(processed),sr)
 assert len(processed)==round(source['source_end_s']*sr)-round(source['source_start_s']*sr)
 assert np.all(processed[mask==0]==0);assert audio_id==result['input_evidence']['audio_identity']
 ownership=read(W/'job_origins'/f"{source['pair']}.json")
 prepared=prepare_input(row,ownership['old_work'],W,cfg['nodes'][node]['resource_work'])
 assert prepared['prompt']==result['prompt'] and prepared['recipe']==recipe
 request_data=request(processor,prepared,'audit-'+sid)
 assert request_data['prompt']==result['rendered_prompt']
 assert request_data['mm_processor_kwargs']['use_audio_in_video'] and not request_data['mm_processor_kwargs']['do_sample_frames']
 assert all(a['input_fingerprint']==fingerprint([prepared['prompt'],prepared['evidence']]) for a in result['attempts'])
 if fingerprint(prepared['evidence'])!=fingerprint(result['input_evidence']):
  diff={k:dict(actual=prepared['evidence'].get(k),saved=result['input_evidence'].get(k)) for k in set(prepared['evidence'])|set(result['input_evidence']) if fingerprint(prepared['evidence'].get(k))!=fingerprint(result['input_evidence'].get(k))}
  write(W/'input_chain_audit_20261031'/f'evidence-diff-{node}.json',diff)
  raise AssertionError(list(diff))
 assert prepared['frames'].shape[-3:]==(512,512,3)
 nod=read(prepared['evidence']['nod_supplement']['detector_file']);assert nod['source']==source and nod['fps']==25
 assert nod['media']['source_start_s']==start and nod['media']['video_transform']==source['video_transform']
 for e in nod['events']:assert 0<=e['start']<e['end']<=duration
 quality=result['visual_quality'];assert quality['decision']=='retain';assert hashlib.sha256(Path(quality['measurement_file']).read_bytes()).hexdigest()==quality['measurement_sha256']
 out.append(dict(sample_id=sid,node=node,source_split=source['split'],source_result=str(srcpath),listener_id=source['listener_id'],speaker_id=source['speaker_id'],crop=source['video_transform']['crop'],duration_s=duration,full_silence=audio_id['full_silence'],outside_mask_samples=int((mask==0).sum()),video_frames=len(prepared['frames']),nod_events=len(nod['events']),old_work=ownership['old_work'],passed=True,seconds=time.time()-tick))
report=dict(at=time.time(),node=node,passed=True,samples=out);write(W/'input_chain_audit_20261031'/f'replay-{node}.json',report);print(json.dumps(report))
