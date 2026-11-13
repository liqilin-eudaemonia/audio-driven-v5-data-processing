from pathlib import Path
import sys,json,subprocess,pickle
import cv2,numpy as np,mediapipe as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import glare_pair_review_backend as src
W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';ID='V00_S1453_I00001162_P1124A';target=pickle.load((W/('reference_'+ID+'_126.007_129.457.pkl')).open('rb'));record=json.loads((W/(ID+'.json')).read_text());t=next(t for t in src.tracks('V00_S1453_I00001162') if t['id']==ID);w,h=record['frame_size']
p=subprocess.Popen(['ffmpeg','-loglevel','error','-ss','126.007','-i',src.uri(t['source']),'-vf','fps=25','-frames:v','12','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE);frames=[]
for i in range(12):
 b=p.stdout.read(w*h*3)
 if len(b)<w*h*3:break
 frames.append(np.frombuffer(b,np.uint8).reshape(h,w,3))
p.stdout.close();p.wait()
model=str(ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task');reports=[]
candidates={'source_first_'+k:v for k,v in record['candidates'].items()}
segment=W/(ID+'_126.007_segment.json')
if segment.exists():candidates.update({'segment_first_'+k:v for k,v in json.loads(segment.read_text())['candidates'].items()})
for name,box in candidates.items():
 opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True,output_facial_transformation_matrixes=True)
 errors=[];bounds=[]
 with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as detector:
  x1,y1,x2,y2=box
  for i,frame in enumerate(frames):
   crop=cv2.resize(frame[y1:y2,x1:x2],(512,512));res=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),i*40)
   if res.face_landmarks and target[i]['detected']:
    a=np.array([[l.x,l.y] for l in res.face_landmarks[0]]);b=np.array(target[i]['landmarks'])[:,:2];errors.append(float(np.sqrt(np.mean((a-b)**2))*512));bounds.append((a.min(0)*512).tolist()+(a.max(0)*512).tolist())
 reports.append({'candidate':name,'box':box,'matched_frames':len(errors),'landmark_xy_rmse_px_mean':float(np.mean(errors)) if errors else None,'first_reconstructed_landmark_bbox':bounds[0] if bounds else None,'published_first_bbox':target[0]['bbox_px']})
 print(reports[-1],flush=True)
result={'reference':ID+'_126.007_129.457.pkl','frames':len(frames),'candidates':reports,'best':min((r for r in reports if r['matched_frames']),key=lambda r:r['landmark_xy_rmse_px_mean'])['candidate'],'note':'Limited sample geometric comparison; not proof of exact training crop.'};(W/'geometry_validation.json').write_text(json.dumps(result,indent=2))
