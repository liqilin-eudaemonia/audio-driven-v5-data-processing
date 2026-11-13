"""GPU7 Face-Alignment crop hypotheses for one published Seamless clip."""
from pathlib import Path
import sys,json,subprocess,argparse,time,os
import cv2,numpy as np,torch,face_alignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import glare_pair_review_backend as src
from reconstruct_glare_crop_math import boxes_for_face
W=ROOT/'evaluation/glare_pair_review/crop_reconstruction';W.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--id',action='append',required=True);a=p.parse_args()
fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False);assert torch.cuda.device_count()==1
for ident in a.id:
 r=src.clip('seamless',ident)
 for t in src.tracks(r['source_id']):
  dest=W/(t['id']+f"_{r['start']:.3f}_segment.json")
  if dest.exists():continue
  meta=src.probe_video(archive=t['source']['archive'],member_name=t['source']['member'],offset=t['source']['offset'],byte_size=t['source']['byte_size']);w,h=meta['width'],meta['height']
  proc=subprocess.Popen(['ffmpeg','-loglevel','error','-threads','2','-ss',str(r['start']),'-i',src.uri(t['source']),'-t',str(r['end']-r['start']),'-an','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);index=0;record=None
  try:
   while True:
    b=proc.stdout.read(w*h*3)
    if len(b)<w*h*3:break
    frame=np.frombuffer(b,np.uint8).reshape(h,w,3);boxes=fa.face_detector.detect_from_image(frame);valid=[b for b in boxes if b[4]>=.5]
    if valid:
     box=max(valid,key=lambda b:b[4]);record={'status':'reconstructed','source_id':t['id'],'scope':'first valid frame of this prepared clip','clip_start':r['start'],'clip_end':r['end'],'first_valid_frame':index,'first_valid_source_time':r['start']+index/meta['fps'],'source_size':[w,h],'source_fps':meta['fps'],'face_box_xyxy':list(map(float,box[:4])),'face_alignment_version':face_alignment.__version__,'detector':'SFD, full-resolution, score >= 0.5','confidence':float(box[4]),'scale':1.5,'candidates':boxes_for_face(box),'selected':'square_1p5','output_size':[512,512],'boundary':'black padding outside source','created':time.time(),'gpu':torch.cuda.get_device_name(),'note':'Most-likely reconstruction, not an author-released crop box.'}
     mult=360/frame.shape[0];small=cv2.resize(frame,(0,0),fx=mult,fy=mult,interpolation=cv2.INTER_AREA);small_boxes=fa.face_detector.detect_from_image(small)
     if len(small_boxes):
      bsmall=max(small_boxes,key=lambda b:b[4]);mapped=[float(v/mult) for v in bsmall[:4]]+[float(bsmall[4])];record['resize360_box_xyxy']=mapped[:4];record['candidates'].update({k+'_resize360':v for k,v in boxes_for_face(mapped).items()});record['selected']='square_1p5_resize360'
     tmp=dest.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(dest);print(t['id'],r['start'],record['candidates'],flush=True);break
    index+=1
  finally:
   proc.terminate();proc.stdout.close()
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  if record is None:raise RuntimeError('No valid face in '+t['id'])
