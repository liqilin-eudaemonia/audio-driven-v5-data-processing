import av,numpy as np,importlib.util,json,time,bisect
from pathlib import Path
from listen_labeling.media import crop_resize
from listen_benchmark.feature_pipeline import source_row
r=Path(__file__).resolve().parent.parent;p=r/'media_export_v2/scripts/export_listen_debug50_media.py';s=importlib.util.spec_from_file_location('fast',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(174);results=[]
for W,H,fmt in [(768,1024,'yuv420p'),(770,1026,'yuv420p'),(2160,3840,'yuv420p'),(641,961,'yuv420p'),(768,1024,'yuvj420p'),(768,1024,'rgb24')]:
 f=av.VideoFrame(W,H,fmt)
 for plane in f.planes:
  a=np.frombuffer(plane,dtype=np.uint8);a[:]=rng.integers(0,256,len(a),dtype=np.uint8)
 for colorspace in [1,5]:
  for color_range in [1,2]:
   f.colorspace=colorspace;f.color_range=color_range
   for x,y,w in [(0,0,384),(1,1,385),(W-385,H-385,385),(33,47,511)]:
    transform=dict(crop=dict(x=x,y=y,width=w,height=w),source_size=[W,H],output_size=[512,512],resample='bicubic')
    old=np.array(crop_resize(f.to_image(),transform));new=np.array(m.crop_frame(f,transform));eq=np.array_equal(old,new)
    results.append(dict(W=W,H=H,format=fmt,x=x,y=y,w=w,colorspace=colorspace,color_range=color_range,equal=eq));assert eq,results[-1]
# Time selection parity for differing frame rates, phase, gaps, and tail reuse.
for rate in [15,24,25,29.97,30,50,60]:
 for phase in [0,.0001,.01,.03]:
  times=[phase+i/rate for i in range(int(5*rate)) if phase+i/rate<4.793708333]
  n=int(np.ceil(4.793708333*25-1e-9));indices=[]
  for k,t in enumerate(times):
   while len(indices)<n and len(indices)/25<=t:indices.append(k)
  indices.extend([len(times)-1]*(n-len(indices)))
  assert indices==[min(bisect.bisect_left(times,i/25),len(times)-1) for i in range(n)]
report=dict(pixel_cases=len(results),all_equal=True,timing_cases=28,results=results)
(r/'media_export_v2/pixel_validation_final.json').write_text(json.dumps(report,indent=2));print('PIXELS/TIMES PASS',len(results),28,flush=True)
a=next(x['candidate'] for x in json.loads((r/'jobs/0874-0011.json').read_text())['samples'] if x['candidate']['sample_id'].endswith('d2904fab11cf'));full=source_row(r,a['source']);sid='profile-'+a['sample_id'];out=r/'audio_grid/media_profile_fast_final';t=time.monotonic()
_,_,meta=m.export_one(dict(sample_id=sid,source=a['source'],duration_s=a['duration_s']),full,out)
old=json.loads((r/'audio_grid/media_profile/media'/sid/'media.json').read_text());report=dict(seconds=time.monotonic()-t,files_equal=old['sha256']==meta['sha256'],times_equal=old['source_frame_times_s']==meta['source_frame_times_s'],indices_equal=old['output_to_source_indices']==meta['output_to_source_indices'],hashes=meta['sha256']);assert report['files_equal'] and report['times_equal'] and report['indices_equal'];print('REAL CLIP PASS',report,flush=True);(r/'media_export_v2/real_clip_validation_final.json').write_text(json.dumps(report,indent=2))
