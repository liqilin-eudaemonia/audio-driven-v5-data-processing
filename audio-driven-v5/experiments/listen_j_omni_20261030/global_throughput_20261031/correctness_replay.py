import sys,json,hashlib,time,math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');W=R/'experiments/listen_j_omni_20261030';sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,result_path,write
from listen_benchmark.full_raw import prepare
from listen_benchmark.media import uniform_video
from listen_benchmark.j_omni import audio_recipe,read_audio,fingerprint
from listen_benchmark.j_omni_crop import effective_row
from listen_benchmark.native_caption_nod import render
cfg=read(W/'config.json');ids=read(W/'global_throughput_20261031/D2_after_roi/finished_ids.json');selected={}
for item in ids:
 r=read(result_path(W,'caption',item['sample_id']));silent=r['input_evidence']['listener_audio_is_silent']
 if silent not in selected:selected[silent]=r
 if len(selected)==2:break
assert len(selected)==2
out=[]
for silent,r in selected.items():
 s=r['source'];sid=r['sample_id'];row=dict(sample_id=sid,source=s)
 actual,prov=effective_row(dict(sample_id=sid,source=r['original_source']),cfg);assert actual['source']==s and prov==r['crop_provenance']
 assert s['listener_id']!=s['speaker_id'] and Path(s['source']['member']).stem==s['listener_id']
 recipe=audio_recipe(row,W);assert recipe==r['audio_recipe']
 wave,sr,identity=read_audio(recipe);assert identity==r['input_evidence']['audio_identity'];audio=wave[:,0]
 if sr!=16000:
  g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g).astype(np.float32)
 assert hashlib.sha256(audio.tobytes()).hexdigest()==r['input_evidence']['audio_encoder_sha256']
 media=prepare(s,'visual');frames,_=uniform_video(media,8)
 assert hashlib.sha256(frames.tobytes()).hexdigest()==r['input_evidence']['video_frames_sha256']
 assert media['meta']['video']==r['input_evidence']['video_evidence']
 nod=r['input_evidence']['nod_supplement'];n=read(Path(nod['detector_file']))
 assert hashlib.sha256(Path(nod['detector_file']).read_bytes()).hexdigest()==nod['detector_sha256']
 expected=r['original_source'] if nod.get('crop_reuse')=='identical_video_transform' else s
 assert n['source']==expected and n['source']['video_transform']==s['video_transform']
 assert all(n['source'][k]==s[k] for k in ['source','source_start_s','source_end_s','duration_s','listener_id'])
 assert all(0<=e['start']<e['end']<=s['duration_s'] for e in nod['events'])
 prompt=render(dict(sample_id=sid,duration_s=s['duration_s'],nod_supplement=nod,audio_preflight=dict(full_silence=identity['full_silence'])),style=cfg['prompt_style']);assert prompt==r['prompt']
 out.append(dict(sample_id=sid,silent=silent,passed=True,checks=['listener_identity','corrected_crop','official_recipe','masked_waveform','16k_audio_hash','legacy_decode_video_pixels','time_grid','nod_source_and_intervals','actual_prompt']))
 print(json.dumps(out[-1]),flush=True)
write(W/'global_throughput_20261031/correctness_replay_20261031.json',dict(at=time.time(),passed=True,samples=out,scope='two production samples, one silent and one audible; no inference rerun'))
