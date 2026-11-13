import sys,json,hashlib
from pathlib import Path
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.full_raw import open_source
from listen_benchmark.frame_crop import crop_original_frame
import numpy as np,av
D=R/'experiments/listen_j_omni_20261030/input_speed_v1_20261031';out=[]
for i in [0,1]:
 row=read(D/'shared_replay'/str(i)/'row.json');s=row['source'];origin=s['source_start_s']
 with open_source(s['source']) as (container,reader):
  st=container.streams.video[0];st.codec_context.thread_count=1;container.seek(int(origin/st.time_base),stream=st,backward=True,any_frame=False)
  for f in container.decode(st):
   if float(f.pts*f.time_base)<origin:continue
   im=np.asarray(crop_original_frame(f,s['video_transform']));np.save(D/f'rgb-{i}-av{av.__version__}.npy',im)
   planes=[]
   for p in f.planes:
    planes.append(dict(width=p.width,height=p.height,line_size=p.line_size,sha256=hashlib.sha256(np.asarray(p).reshape(p.height,p.line_size)[:,:p.width*(2 if '10' in f.format.name else 1)].copy()).hexdigest()))
   out.append(dict(sample=row['sample_id'],pts=f.pts,format=f.format.name,color_range=f.color_range,colorspace=f.colorspace,planes=planes,rgb_sha=hashlib.sha256(im).hexdigest()));break
write(D/f'pixel-av{av.__version__}.json',dict(av=av.__version__,libraries=av.library_versions,samples=out));print(av.__version__,[(x['format'],x['rgb_sha']) for x in out])
