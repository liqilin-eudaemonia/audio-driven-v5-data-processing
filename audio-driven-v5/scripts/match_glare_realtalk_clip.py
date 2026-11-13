"""Approximate clip localization by released pose/blendshape temporal fingerprint."""
from pathlib import Path
import json,pickle,subprocess,time
import cv2,numpy as np,mediapipe as mp
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';target=pickle.load((W/'reference_-1wCGAxqT_4_0003.pkl').open('rb'));keys=sorted(target[0]['blendshapes']);model=str(ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task')
def options():return mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True,output_facial_transformation_matrixes=True)
def vector(frame,detector,stamp):
 r=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=frame),stamp)
 if not r.face_landmarks:return np.full(len(keys),np.nan)
 b={v.category_name:v.score for v in r.face_blendshapes[0]};return np.array([b.get(k,0) for k in keys])
rate=5;cache=W/'realtalk_5fps_fingerprints.npz'
if cache.exists():data=np.load(cache);values=[data['p0'],data['p1']]
else:
 boxes=[json.loads((W/f'realtalk_-1wCGAxqT_4_p{i}.json').read_text())['candidates']['square_1p5'] for i in range(2)];dets=[mp.tasks.vision.FaceLandmarker.create_from_options(options()) for _ in range(2)];values=[[],[]]
 p=subprocess.Popen(['ffmpeg','-loglevel','error','-i','/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4.avi','-vf',f'fps={rate}','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE);n=0
 try:
  while True:
   b=p.stdout.read(1280*720*3)
   if len(b)<1280*720*3:break
   frame=np.frombuffer(b,np.uint8).reshape(720,1280,3)
   for i,(box,det) in enumerate(zip(boxes,dets)):
    x1,y1,x2,y2=box;crop=cv2.resize(frame[y1:y2,x1:x2],(512,512));values[i].append(vector(crop,det,n*200))
   n+=1
   if n%250==0:print('scan seconds',n/rate,flush=True);(W/'realtalk_match_progress.json').write_text(json.dumps({'frames':n,'source_seconds':n/rate,'updated':time.time()}))
 finally:
  p.stdout.close();p.wait()
  for d in dets:d.close()
 values=[np.array(v) for v in values];np.savez_compressed(cache,p0=values[0],p1=values[1],keys=np.array(keys))
t=np.array([[f['blendshapes'].get(k,0) for k in keys] for f in target[::5]]);active=np.std(t,axis=0)>.015;scale=np.maximum(np.std(t,axis=0),.05);n=len(t);scores=[]
for person,v in enumerate(values):
 for start in range(len(v)-n+1):
  win=v[start:start+n]
  if not np.isfinite(win).all():continue
  dynamics=((win-win.mean(0))-(t-t.mean(0)))/scale
  score=np.mean(dynamics[:,active]**2)+.15*np.mean(((win.mean(0)-t.mean(0))/.15)**2)
  scores.append({'person':f'p{person}','start':start/rate,'end':start/rate+len(target)/25,'score':float(score)})
scores.sort(key=lambda x:x['score']);separate=[]
for row in scores:
 if all(row['person']!=r['person'] or abs(row['start']-r['start'])>8 for r in separate):separate.append(row)
 if len(separate)==8:break
result={'method':'Published 52 blendshape temporal sequence, 5fps coarse sliding-window matching','candidates':separate,'active_features':[k for k,a in zip(keys,active) if a],'status':'candidate_not_verified','note':'Not an author-supplied source time; requires fine-grained verification before being presented as corresponding frames.'};(W/'realtalk_match_candidates.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
