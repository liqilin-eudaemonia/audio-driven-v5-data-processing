import sys,os,time,json,inspect,hashlib
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark import full_raw
from listen_benchmark.frame_crop import crop_original_frame,crop_original_frame_reused
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni import fingerprint
from listen_benchmark.media import uniform_video
W=R/'experiments/listen_j_omni_20261030';D=W/'performance_review_20261031'
text=inspect.getsource(full_raw.prepare);needle="im=crop_resize(frame.to_image(),row['video_transform'])";assert text.count(needle)==1
funcs={'original':full_raw.prepare}
for name,f in [('crop_only',crop_original_frame),('reuse_converter',crop_original_frame_reused)]:
 env=dict(full_raw.__dict__,candidate_crop=f);exec(text.replace(needle,"im=candidate_crop(frame,row['video_transform'])"),env);funcs[name]=env['prepare']
items=read(W/'corrected_crop_rollout_20261031/deployment_audit.json')['new_samples'];selected=[];sources=set()
for i in items:
 r=read(i['path']);key=r['source']['source']['member']
 if key in sources:continue
 sources.add(key);selected.append(r)
 if len(selected)==4:break
out=[]
for k,r in enumerate(selected):
 comparisons={};order=['original','crop_only','reuse_converter'] if k%2==0 else ['reuse_converter','crop_only','original']
 for name in order:
  t=time.monotonic();m=funcs[name](r['source'],'visual');frames,ev=uniform_video(m,8)
  comparisons[name]=dict(seconds=time.monotonic()-t,frame_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),metadata_sha256=fingerprint(m['meta']),frames=len(frames))
  print(r['sample_id'],name,round(comparisons[name]['seconds'],3),flush=True)
 assert len({v['frame_sha256'] for v in comparisons.values()})==1
 assert len({v['metadata_sha256'] for v in comparisons.values()})==1
 out.append(dict(sample_id=r['sample_id'],source_size=r['source']['video_transform']['source_size'],duration=r['source']['duration_s'],order=order,comparisons=comparisons))
 write(D/'crop_benchmark.json',dict(at=time.time(),completed=len(out),passed=True,samples=out))
print('DONE',flush=True)
