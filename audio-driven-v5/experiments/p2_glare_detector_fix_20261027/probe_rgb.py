import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_ENABLE_ONEDNN_OPTS']='0';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import cv2,torch,tensorflow as tf
from fer import FER
torch.set_num_threads(2);tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
OUT=Path(__file__).resolve().parent;model=FER(mtcnn=True);results=[]
for item in json.loads((OUT/'probe_inputs.json').read_text()):
 cap=cv2.VideoCapture(item['video']);frames=[]
 for i in range(max(item['indexes'])+1):
  ok,bgr=cap.read();assert ok
  if i not in item['indexes']:continue
  old=model.find_faces(bgr);correct=model.find_faces(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));emotions=model.detect_emotions(bgr,face_rectangles=correct)
  frames.append(dict(frame_index=i,bgr_faces=len(old),rgb_faces=len(correct),rgb_emotions=len(emotions)))
 cap.release();results.append(dict(sample_id=item['sample_id'],side=item['side'],video=item['video'],frames=frames));print(results[-1],flush=True)
(OUT/'rgb_probe.json').write_text(json.dumps(results,indent=2));print('recovered',sum(f['bgr_faces']==0 and f['rgb_emotions']==1 for r in results for f in r['frames']),'of',sum(len(r['frames']) for r in results))
