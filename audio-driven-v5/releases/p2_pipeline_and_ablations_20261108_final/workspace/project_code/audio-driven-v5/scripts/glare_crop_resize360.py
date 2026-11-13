from pathlib import Path
import sys,json,subprocess,os
import cv2,numpy as np,face_alignment,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import glare_pair_review_backend as src
from reconstruct_glare_crop_math import boxes_for_face
W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
for p in W.glob('V*.json'):
 d=json.loads(p.read_text())
 if not isinstance(d,dict) or 'candidates' not in d or 'source_id' not in d:continue
 if 'resize360_box_xyxy' in d:continue
 pair=d['source_id'].rsplit('_',1)[0];t=next(t for t in src.tracks(pair) if t['id']==d['source_id'])
 b=subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(d.get('first_valid_source_time',d.get('first_valid_time',0))),'-i',src.uri(t['source']),'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','2','pipe:1'])
 frame=cv2.cvtColor(cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB);mult=360/frame.shape[0];small=cv2.resize(frame,(0,0),fx=mult,fy=mult,interpolation=cv2.INTER_AREA)
 boxes=fa.face_detector.detect_from_image(small);box=max(boxes,key=lambda x:x[4]);mapped=[float(x/mult) for x in box[:4]]+[float(box[4])]
 d['resize360_box_xyxy']=mapped[:4];d['resize360_confidence']=mapped[4];d['resize360_note']='Alternative detector input scaling, following public GLARE generate.py image preprocessing; not specified in training appendix.'
 d['candidates'].update({k+'_resize360':v for k,v in boxes_for_face(mapped).items()});d['selected']='square_1p5_resize360' if p.name.endswith('_segment.json') else d.get('selected','square_1p5');tmp=p.with_suffix('.resize.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p);print(p.name,d['candidates'],flush=True)
