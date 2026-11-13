"""Reconstruct likely GLARE crops using Face-Alignment; run only on node8 GPU7."""
from pathlib import Path
import sys,json,subprocess,math,time
import numpy as np,cv2,torch,face_alignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import glare_pair_review_backend as src
WORK=ROOT/'evaluation/glare_pair_review/crop_reconstruction';WORK.mkdir(exist_ok=True)
OUT=ROOT/'data_previews/glare_reaction_annotation/media/reconstructed';OUT.mkdir(exist_ok=True)
def candidate(box,w,h):
 x1,y1,x2,y2=box[:4];cx=(x1+x2)/2;cy=(y1+y2)/2;side=int(round(max(x2-x1,y2-y1)*1.5));side+=side%2
 square=[int(round(cx-side/2)),int(round(cy-side/2)),0,0];square[2]=square[0]+side;square[3]=square[1]+side
 rect=[int(round(cx-(x2-x1)*.75)),int(round(cy-(y2-y1)*.75)),int(round(cx+(x2-x1)*.75)),int(round(cy+(y2-y1)*.75))]
 return {'square_1p5':square,'rectangle_1p5':rect}
fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
assert torch.cuda.device_count()==1
print('FA',face_alignment.__version__,'GPU',torch.cuda.get_device_name(),flush=True)
for pair in ['V00_S0058_I00000536','V00_S1127_I00001166','V00_S0044_I00000309','V00_S1453_I00001162']:
 for t in src.tracks(pair):
  out=WORK/(t['id']+'.json')
  if out.exists():continue
  meta=src.probe_video(archive=t['source']['archive'],member_name=t['source']['member'],offset=t['source']['offset'],byte_size=t['source']['byte_size']);w,h=meta['width'],meta['height']
  p=subprocess.Popen(['ffmpeg','-loglevel','error','-threads','2','-i',src.uri(t['source']),'-an','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
  index=0;record=None
  try:
   while True:
    b=p.stdout.read(w*h*3)
    if len(b)!=w*h*3:break
    frame=np.frombuffer(b,dtype=np.uint8).reshape(h,w,3);boxes=fa.face_detector.detect_from_image(frame);valid=[b for b in boxes if len(b)<5 or b[4]>=.5]
    if valid:
     box=max(valid,key=lambda b:b[4] if len(b)>4 else 1);candidates=candidate(box,w,h)
     image=OUT/(t['id']+'_first_frame.png');cv2.imwrite(str(image),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
     record={'source_id':t['id'],'source':t['source'],'frame_size':[w,h],'source_fps':meta['fps'],'first_valid_frame':index,'first_valid_time':index/meta['fps'],'face_alignment_version':face_alignment.__version__,'detector':'SFD, default full-resolution input; confidence >= 0.5','detected_box_xyxy':list(map(float,box[:4])),'detection_confidence':float(box[4]),'candidates':candidates,'selected':'square_1p5','inference':'Reconstruction, not a recovered original training crop. Paper specifies 1.5x; squaring, pixel rounding and border padding are explicit implementation assumptions.','padding':'black if beyond source boundary','output_size':[512,512],'gpu':torch.cuda.get_device_name(),'created':time.time()}
     out.write_text(json.dumps(record,indent=2));print(t['id'],index,record['detected_box_xyxy'],candidates,flush=True);break
    index+=1
  finally:
   p.terminate();p.stdout.close()
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  if record is None:print('NO FACE',t['id'],flush=True)
print('CROP BOXES COMPLETE',flush=True)
p=Path('/private/liqi_workspace/realtalk_inspection/-1wCGAxqT_4.avi')
if p.exists() and not (WORK/'realtalk_-1wCGAxqT_4_p0.json').exists():
 cap=cv2.VideoCapture(str(p));i=0
 while True:
  ok,bgr=cap.read()
  if not ok:break
  boxes=fa.face_detector.detect_from_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));valid=sorted(sorted([b for b in boxes if b[4]>=.5],key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)[:2],key=lambda b:b[0])
  if len(valid)>=2:
   h,w=bgr.shape[:2]
   for j,b in enumerate(valid[:2]):
    ident=f'realtalk_-1wCGAxqT_4_p{j}';rec={'source_id':ident,'source_path':str(p),'frame_size':[w,h],'source_fps':cap.get(cv2.CAP_PROP_FPS),'first_valid_frame':i,'first_valid_time':i/cap.get(cv2.CAP_PROP_FPS),'detected_box_xyxy':list(map(float,b[:4])),'detection_confidence':float(b[4]),'candidates':candidate(b,w,h),'selected':'square_1p5','participant_assumption':'left-to-right p0/p1; original training identity not released','face_alignment_version':face_alignment.__version__}
    (WORK/(ident+'.json')).write_text(json.dumps(rec,indent=2));print('REALTALK',ident,rec,flush=True)
   break
  i+=1
 cap.release()
