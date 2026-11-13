from pathlib import Path
import pickle,json,subprocess
import cv2,numpy as np,mediapipe as mp
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';target=pickle.load((W/'reference_-1wCGAxqT_4_0003.pkl').open('rb'));start=json.loads((W/'realtalk_localization.json').read_text())['best']['start'];model=str(ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task')
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-ss',str(start),'-i','/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4.avi','-frames:v','20','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE);frames=[]
for i in range(20):
 b=proc.stdout.read(1280*720*3)
 if len(b)<1280*720*3:break
 frames.append(np.frombuffer(b,np.uint8).reshape(720,1280,3))
proc.stdout.close();proc.wait();reports=[]
for x,y in [(3,103),(4,104),(3,104),(4,103)]:
 for side in [364,365,366]:
  options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1)
  errors=[]
  with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
   for i,frame in enumerate(frames):
    crop=cv2.resize(frame[y:y+side,x:x+side],(512,512));r=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),i*40)
    if r.face_landmarks:
     a=np.array([[p.x,p.y] for p in r.face_landmarks[0]]);b=np.array(target[i]['landmarks'])[:,:2];errors.append(float(np.sqrt(np.mean((a-b)**2))*512))
  reports.append({'box_xyxy':[x,y,x+side,y+side],'mean_landmark_rmse_px':float(np.mean(errors)),'frames':len(errors)})
reports.sort(key=lambda x:x['mean_landmark_rmse_px']);result={'best':reports[0],'alternatives':reports[1:],'note':'20-frame direct landmark verification of an inferred crop, not an author-supplied original crop.'};(W/'realtalk_crop_fit_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
p=W/'realtalk_crop_inferred.json';d=json.loads(p.read_text());d['box_xyxy']=reports[0]['box_xyxy'];d['direct_reextraction_validation']=reports[0];p.write_text(json.dumps(d,indent=2))
