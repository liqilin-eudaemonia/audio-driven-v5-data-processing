import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');OUT=ROOT/'experiments/glare_cache_28gpu_20261110/fer_gpu_probe'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if sys.argv[1]=='decode':
 from glare_shared_decode import decode_group,production_pixels
 with (ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection/selected.jsonl').open() as f:rows=[json.loads(next(f)) for _ in range(3)]
 paths=[]
 for i,row in enumerate(rows):
  d=OUT/('input%d'%i);d.mkdir(exist_ok=True)
  info=decode_group([row],d)
  folder=Path(info['items'][0]['folder']);pixels,meta=production_pixels(folder);paths.append(str(folder/'detector_bgr.npy'))
 (OUT/'inputs.json').write_text(json.dumps(paths));print('decoded',len(paths))
else:
 import numpy as np,torch,cv2,tensorflow as tf
 torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
 tf.config.set_visible_devices([], 'GPU')
 from fer.fer import FER
 model=FER(mtcnn=True,use_tflite=True)
 assert ('cuda' in str(model._mtcnn.device))==sys.argv[1].startswith('gpu')
 paths=json.loads((OUT/'inputs.json').read_text());rows=[]
 model.detect_emotions(np.load(paths[0],mmap_mode='r')[0])
 for path in paths:
  pixels=np.load(path,mmap_mode='r');start=time.monotonic()
  if sys.argv[1].startswith('gpu_batch'):
   faces=[]
   for offset in range(0,len(pixels),32):
    batch=np.array(pixels[offset:offset+32]);boxes,_=model._mtcnn.detect(batch)
    for frame,bb in zip(batch,boxes):
     rects=[] if bb is None else [[int(f[0]),int(f[1]),int(f[2])-int(f[0]),int(f[3])-int(f[1])] for f in bb]
     faces.append(model.detect_emotions(frame,face_rectangles=rects))
  else:faces=[model.detect_emotions(x) for x in pixels]
  if sys.argv[1].startswith('gpu'):torch.cuda.synchronize()
  rows.append(dict(path=path,frames=len(pixels),seconds=time.monotonic()-start,faces=faces))
 (OUT/(sys.argv[1]+'.json')).write_text(json.dumps(rows));print(json.dumps([dict(frames=r['frames'],seconds=r['seconds']) for r in rows]))
