from pathlib import Path
import json,pickle,subprocess
import cv2,numpy as np,mediapipe as mp
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';target=pickle.load((W/'reference_-1wCGAxqT_4_0003.pkl').open('rb'));keys=sorted(target[0]['blendshapes']);model=str(ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task');candidates=json.loads((W/'realtalk_match_candidates.json').read_text())['candidates'][:3]
t=np.array([[r['blendshapes'].get(k,0) for k in keys] for r in target]);active=t.std(0)>.015;scale=np.maximum(t.std(0),.05);reports=[]
for candidate in candidates:
 person=candidate['person'];box=json.loads((W/f'realtalk_-1wCGAxqT_4_{person}.json').read_text())['candidates']['square_1p5'];start=round((candidate['start']-1.2)*25)/25;end=candidate['end']+1.2
 options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True)
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-ss',str(start),'-i','/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4.avi','-t',str(end-start),'-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE);values=[];landmarks=[]
 with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
  i=0
  while True:
   b=proc.stdout.read(1280*720*3)
   if len(b)<1280*720*3:break
   f=np.frombuffer(b,np.uint8).reshape(720,1280,3);x1,y1,x2,y2=box;crop=cv2.resize(f[y1:y2,x1:x2],(512,512));r=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),i*40)
   if r.face_landmarks:
    d={v.category_name:v.score for v in r.face_blendshapes[0]};values.append([d.get(k,0) for k in keys]);landmarks.append([[v.x,v.y] for v in r.face_landmarks[0]])
   else:values.append([np.nan]*len(keys));landmarks.append([[np.nan]*2]*478)
   i+=1
 proc.stdout.close();proc.wait();values=np.array(values);best=None
 for j in range(len(values)-len(t)+1):
  v=values[j:j+len(t)]
  if not np.isfinite(v).all():continue
  dynamic=((v-v.mean(0))-(t-t.mean(0)))/scale;score=np.mean(dynamic[3:,active]**2)+.15*np.mean(((v.mean(0)-t.mean(0))/.15)**2)
  if best is None or score<best['score']:best={'person':person,'start':start+j/25,'end':start+j/25+len(t)/25,'score':float(score),'offset':j}
 if best:
  idx=best['offset'];np.savez_compressed(W/f"realtalk_refine_{person}_{candidate['start']}.npz",landmarks=np.array(landmarks)[idx:idx+len(t)],blendshapes=values[idx:idx+len(t)],target=t,keys=np.array(keys));reports.append(best);print(best,flush=True)
reports.sort(key=lambda x:x['score']);result={'status':'inferred_from_published_temporal_fingerprint','target':'-1wCGAxqT_4/0003.pkl','frames':len(t),'fps':25,'best':reports[0],'alternatives':reports[1:],'best_vs_next_ratio':reports[1]['score']/reports[0]['score'],'note':'Inferred source interval, not an author-released mapping; independently cross-check with original active-speaker annotations.'};(W/'realtalk_localization.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
