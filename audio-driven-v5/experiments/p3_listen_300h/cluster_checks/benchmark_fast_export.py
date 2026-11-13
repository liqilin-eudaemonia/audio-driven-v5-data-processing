import json,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from cache_listen_corpus import configure_media_export
from listen_fast_media import export
from p3_listen import media
from p3_listen.common import read,write
import numpy as np
import soundfile as sf
from PIL import Image
root=ROOT/'experiments/p3_listen_300h';out=root/'cluster_checks/fast_export_check'
ids={r['sample_id'] for r in read(root/'cluster_checks/cache_timing.json')['rows']}
ids.add('naturalistic_V02_S5109_I00000302_clip_0005_sub00_w001')
rows=[r for r in read(root/'windows.json')['windows'] if r['sample_id'] in ids]
configure_media_export(media);checks=[]
for row in rows:
 media.paths(out,row)['marker'].unlink(missing_ok=True)
 begin=time.monotonic();export(out,row);p,_=media.verify_media(out,row);old=media.paths(root,row)
 for key in ('speaker_audio','target_audio'):
  a,sr=sf.read(p[key],dtype='int32');b,sr0=sf.read(old[key],dtype='int32')
  if sr!=sr0 or not np.array_equal(a,b):raise ValueError(row['sample_id']+' '+key+' differs')
 if not np.array_equal(np.asarray(Image.open(p['listener_image'])),np.asarray(Image.open(old['listener_image']))):raise ValueError('reference pixels differ')
 if media.sha(p['target_video'])!=media.sha(old['target_video']):raise ValueError('video bytes differ')
 checks.append(dict(sample_id=row['sample_id'],seconds=time.monotonic()-begin,audio_samples_equal=True,reference_pixels_equal=True,video_bytes_equal=True));print(json.dumps(checks[-1]),flush=True)
write(root/'cluster_checks/fast_export_equivalence.json',dict(checks=checks,all_equal=True))
