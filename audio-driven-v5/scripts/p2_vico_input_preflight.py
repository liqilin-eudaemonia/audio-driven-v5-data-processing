"""Freeze ViCo test43 sources and CPU-only source-conditioned inputs."""
from pathlib import Path
import json,hashlib,math,subprocess,concurrent.futures,io
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'experiments/p2_vico256_combined8000_20261106'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run():
 import av,numpy as np,soundfile as sf
 from PIL import Image
 from scipy.signal import resample_poly
 index=json.loads((ROOT/'evaluation/vico/index.json').read_text());items=[x for x in index['clips'] if x['split']=='test'];assert len(items)==43 and len({x['uuid'] for x in items})==43
 probe={x['id']:x for x in json.loads((WORK/'media_probe.json').read_text())};reports=[]
 def one(r):
  sid=r['uuid'];v=next(x for x in probe[sid]['listener']['streams'] if x['codec_type']=='video');duration=int(v['nb_frames'])/30;assert v['avg_frame_rate']=='30/1' and (v['width'],v['height'])==(256,256)
  folder=WORK/'conditions/vico'/sid;folder.mkdir(parents=True,exist_ok=True)
  with av.open(r['listener']) as c:
   st=c.streams.video[0];st.codec_context.thread_count=1;st.thread_type='SLICE';frame=next(c.decode(video=0));image=frame.to_image().convert('RGB')
  assert image.size==(256,256);image.resize((512,512),Image.Resampling.BICUBIC).save(folder/'first_frame.png')
  with sf.SoundFile(r['audio']) as a:
   rate=a.samplerate;samples=len(a);wave=a.read(dtype='float32',always_2d=True).mean(1)
  common=math.gcd(rate,16000);wave=resample_poly(wave,16000//common,rate//common).astype(np.float32);count=round(duration*16000);delta=count-len(wave);assert abs(delta)<=1600,(sid,delta)
  wave=np.pad(wave,(0,max(0,delta)))[:count];assert len(wave)==count and np.isfinite(wave).all()
  audit=dict(id=sid,speaker_audio_path=r['audio'],speaker_audio_sha256=sha(r['audio']),speaker_first_sample=0,speaker_last_sample=samples,speaker_source_rate=rate,speaker_resample_rate=16000,endpoint_adjustment_samples=delta,endpoint_policy='Trim/pad tail only to original GT frame-grid duration; no time stretching',waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),reference_first_frame=r['listener'],reference_source_sha256=sha(r['listener']),source_frame_pts=frame.pts,source_frame_time=float(frame.time or 0),decoded_listener_frames=1,source_only=True,target_description_in_model_input=False,listener_future_media_in_model_input=False,reference_resize='PIL bicubic 256->512',original_gt=dict(width=256,height=256,fps=30,frames=int(v['nb_frames']),duration_s=duration))
  (folder/'input_audit.json').write_text(json.dumps(audit,indent=2));return audit
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(one,items))
 (WORK/'INPUT_PREFLIGHT.json').write_text(json.dumps(dict(samples=43,source_only=True,listener_decoded_frames_per_sample=1,conditioning_size=[512,512],gt_size=[256,256],gt_fps=30,seconds=sum(r['original_gt']['duration_s'] for r in reports),largest_tail_adjustment_samples=max(abs(r['endpoint_adjustment_samples']) for r in reports),samples_audit=reports),indent=2));print('CPU input preflight passed:',len(reports),'clips; max tail adjustment',max(abs(r['endpoint_adjustment_samples']) for r in reports),'samples @ 16kHz',flush=True)
if __name__=='__main__':run()
