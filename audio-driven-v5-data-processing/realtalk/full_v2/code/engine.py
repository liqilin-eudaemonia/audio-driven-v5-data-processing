"""Persistent models and geometry helpers. Canonical pairs are never relabelled."""
from pathlib import Path
import os,sys,json,hashlib,types,subprocess,math,contextlib
import numpy as np,cv2,soundfile as sf,torch,yaml
from scipy.signal import medfilt
R=Path(__file__).resolve().parents[1];BASE=R.parent;SYNC=Path('/private/codes/joint_gen/complex_motion_train/syncnet_python');AV=BASE/'av_listener_trial_20261024';VERSION='full-listener-sync-boundary-v2'
sys.path.insert(0,str(SYNC));sys.path.insert(0,str(AV/'upstream/clearvoice'))
from SyncNetInstance import SyncNetInstance
from detectors import S3FD
from clearvoice.models.av_mossformer2_tse.av_mossformer2 import AV_MossFormer2_TSE_16K
from clearvoice.utils.decode import decode_one_audio_AV_MossFormer2_TSE_16K
from calibrate import rms,db,scale
from gate import make_gain

def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v) if isinstance(v,Path) else (_ for _ in ()).throw(TypeError(type(v).__name__)))+'\n');tmp.replace(path)
def ff(args,**kw):return subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1',*map(str,args)],check=True,**kw)
def valid_box(b,w,h):
 return bool(np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1] and 0<=(b[0]+b[2])/2<w and 0<=(b[1]+b[3])/2<h)
def iou(a,b):
 d=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));z=float(np.prod(d));return z/max(1e-9,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-z))
def clipbox(b,w,h):
 b=np.array(b,dtype=float).copy();b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h);return b

def context_interval(p,edges):
 a,b=p['start_frame'],p['end_frame'];lo,hi=edges[p['shot']-1:p['shot']+1]
 assert lo<=a<b<=hi
 role_lo,role_hi=p['continuous_interval'];lo=max(lo,role_lo);hi=min(hi,role_hi)
 assert lo<=a<b<=hi
 if b-a>50:return a,b
 length=min(60,hi-lo);st=max(lo,min(a-(length-(b-a))//2,hi-length));return min(st,a),max(b,st+length)

def recover_crop(track,other,w,h):
 med=np.median(track,axis=0);left,top,right,bottom=0,0,w,h
 if other is not None:
  q=np.asarray(other);cx,cy=(med[:2]+med[2:])/2;ox,oy=(q[:2]+q[2:])/2
  gx=abs(cx-ox)-(med[2]-med[0]+q[2]-q[0])/2;gy=abs(cy-oy)-(med[3]-med[1]+q[3]-q[1])/2
  if max(gx,gy)<0:return None
  if gx>=gy:
   if cx<ox:right=int((med[2]+q[0])/2)
   else:left=math.ceil((q[2]+med[0])/2)
  else:
   if cy<oy:bottom=int((med[3]+q[1])/2)
   else:top=math.ceil((q[3]+med[1])/2)
 edge=int(min(math.ceil(max(med[2]-med[0],med[3]-med[1])*2.35),right-left,bottom-top))//2*2
 if edge<2:return None
 x=int(np.clip((med[0]+med[2])/2-edge/2,left,right-edge));y=int(np.clip((med[1]+med[3])/2-edge*.6,top,bottom-edge))
 inside=(track[:,0]>=x)&(track[:,1]>=y)&(track[:,2]<=x+edge)&(track[:,3]<=y+edge)
 if np.mean(inside)<.95:return None
 return {'x':x,'y':y,'width':edge,'height':edge,'native_edge':edge,'method':'identity_anchored_detection_and_horizontal_or_vertical_partition','status':'recovered_geometry_candidate','full_face_inside_fraction':float(np.mean(inside))}

class Engine:
 def __init__(self,tmp):
  torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0);np.random.seed(0);self.tmp=Path(tmp);self.tmp.mkdir(parents=True,exist_ok=True);self.device='cuda' if torch.cuda.is_available() else 'cpu'
  def ns(d):return types.SimpleNamespace(**{k:ns(v) if isinstance(v,dict) else v for k,v in d.items()})
  self.args=ns(yaml.safe_load((AV/'upstream/clearvoice/clearvoice/config/inference/AV_MossFormer2_TSE_16K.yaml').read_text()));self.args.device=torch.device(self.device);self.args.one_time_decode_length=100
  self.model_fingerprints={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in {'separation':AV/'model/last_best_checkpoint.pt','syncnet':SYNC/'data/syncnet_v2.model','detector':SYNC/'detectors/s3fd/weights/sfd_face.pth'}.items()};assert self.model_fingerprints['separation']=='0e60209aab0b31c38deea4ab116100acbbf3a07db8bc559f796f118da1a767d5'
  self.av=AV_MossFormer2_TSE_16K(self.args).model;state=torch.load(AV/'model/last_best_checkpoint.pt',map_location='cpu',weights_only=True)['model'];state={k.removeprefix('module.'):v for k,v in state.items()};self.av.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in self.av.state_dict().items());self.av.to(self.device).eval();del state
  self.sync=SyncNetInstance(device=self.device);self.sync.loadParameters(str(SYNC/'data/syncnet_v2.model'))
  cwd=os.getcwd()
  try:os.chdir(SYNC);self.detector=S3FD(device=self.device)
  finally:os.chdir(cwd)
  self.det_cache={}
 def detect(self,frame,index):
  if index not in self.det_cache:
   rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);boxes=self.detector.detect_faces(rgb,conf_th=.65,scales=[.25])
   if not len(boxes):boxes=self.detector.detect_faces(rgb,conf_th=.65,scales=[.5])
   self.det_cache[index]=boxes[:,:4]
  return self.det_cache[index]
 def track(self,person,lo,hi,ring,boxes,edges,shot,w,h):
  p=int(person[1]);start,end=edges[shot-1:shot+1];raw=boxes[lo:hi,p].copy();ok=np.array([valid_box(b,w,h) for b in raw]);anchor_pool=boxes[max(start,lo-50):min(end,hi+50),p];anchor_pool=np.array([b for b in anchor_pool if valid_box(b,w,h)])
  if not len(anchor_pool):raise ValueError('identity_unresolved:no_valid_same_shot_anchor')
  fallback=np.median(anchor_pool,axis=0);indices=sorted(set([lo,(lo+hi-1)//2,hi-1]+[lo+j for j in np.flatnonzero(~ok)]));matches=[]
  for idx in indices:
   anchor=raw[idx-lo] if ok[idx-lo] else fallback;dets=self.detect(ring[idx],idx);rank=[];other=boxes[idx,1-p]
   for det in dets:
    own=iou(clipbox(anchor,w,h),det);opp=iou(clipbox(other,w,h),det) if valid_box(other,w,h) else 0
    if own>=.1 and own>=opp+.05:rank.append((own,det))
   rank.sort(key=lambda x:-x[0])
   if rank and (len(rank)==1 or rank[0][0]-rank[1][0]>=.05):
    matches.append(idx)
    if not ok[idx-lo]:raw[idx-lo]=rank[0][1];ok[idx-lo]=True
  if not matches:raise ValueError('identity_unresolved:detection_not_bound_to_anchor')
  good=np.flatnonzero(ok)
  if not len(good):raise ValueError('track_unresolved:no_valid_boxes')
  # Never bridge a long disappearance or extrapolate beyond 0.8 sec.
  missing=np.flatnonzero(~ok)
  for j in missing:
   before=good[good<j];after=good[good>j]
   if len(before) and len(after):
    if after[0]-before[-1]>50:raise ValueError('track_unresolved:long_occlusion')
   elif min(abs(good-j))>20:raise ValueError('track_unresolved:unsupported_edge')
  for k in range(4):raw[:,k]=np.interp(np.arange(len(raw)),good,raw[good,k])
  raw=np.array([clipbox(b,w,h) for b in raw]);assert all(valid_box(b,w,h) for b in raw)
  return raw,{'detected_anchor_frames':matches,'interpolated_frames':int(len(missing)),'context_start_frame':lo,'context_end_frame':hi,'person':person,'identity_basis':'original_person_bbox_iou_margin; never speaker activity or left/right'}
 def faces(self,ring,lo,track,path):
  n=len(track);kernel=min(13,n if n%2 else n-1);kernel=max(1,kernel);bs=medfilt(np.maximum(track[:,2]-track[:,0],track[:,3]-track[:,1])/2,kernel);cx=medfilt((track[:,0]+track[:,2])/2,kernel);cy=medfilt((track[:,1]+track[:,3])/2,kernel);out=[]
  for j in range(n):
   s=max(1,float(bs[j]));pad=math.ceil(s*1.8);im=np.pad(ring[lo+j],((pad,pad),(pad,pad),(0,0)),constant_values=110);x=cx[j]+pad;y=cy[j]+pad;face=im[int(y-s):int(y+s*1.8),int(x-s*1.4):int(x+s*1.4)];assert face.size;out.append(cv2.resize(face,(224,224)))
  self.write_face(out,path);cap=cv2.VideoCapture(str(path));v=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   v.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[56:168,56:168])
  cap.release();assert len(v)==n;return out,np.asarray(v)
 def write_face(self,faces,path):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'XVID'),25,(224,224));assert writer.isOpened()
  for im in faces:writer.write(im)
  writer.release()
 def separate(self,mix,visual):
  peak=float(np.max(np.abs(mix)));n=len(mix);frames=len(visual);assert n==frames*640
  if peak==0:return np.zeros(n,np.float32),0
  pad=max(0,13-frames)
  if pad:mix=np.pad(mix,(0,pad*640));visual=np.concatenate([visual,np.repeat(visual[-1:],pad,axis=0)])
  x=(visual.astype(np.float32)/255-.4161)/.1688
  with torch.inference_mode(),open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):a=decode_one_audio_AV_MossFormer2_TSE_16K(self.av,((mix/peak)[None].astype(np.float32),x[None]),self.args)
  a=(a[:n]*peak).astype(np.float32);assert len(a)==n and np.isfinite(a).all();return a,pad
 def score(self,face,a,key,gain=1):
  last=None
  for attempt in range(2):
   try:return self._score(face,a,key,gain)
   except Exception as exc:last=str(exc)
  return {'status':'scoring_error','error':last,'robust_pass':False,'short_candidate':False}
 def _score(self,face,a,key,gain=1):
  n=len(a)//640
  if n<6:return {'status':'insufficient_pair_frames','frames':n,'embedding_count':max(0,n-5),'robust_pass':False,'short_candidate':False}
  wav=self.tmp/f'{key}.wav';sf.write(wav,a*gain,16000,subtype='PCM_16');pcm,_=sf.read(wav)
  if np.all(pcm==0):return {'status':'silent_at_pcm16_precision','frames':n,'robust_pass':False,'short_candidate':False}
  video=self.tmp/f'{key}.avi';ff(['-i',face,'-i',wav,'-map','0:v:0','-map','1:a:0','-c','copy',video],stdout=subprocess.DEVNULL)
  try:
   offset,conf,dist=self.sync.evaluate(types.SimpleNamespace(tmp_dir=str(self.tmp/'sync_tmp'),reference=key,batch_size=20,vshift=15),str(video));conf=float(conf);offset=int(offset)
   if not np.isfinite(conf):raise ValueError('nonfinite_sync_confidence')
   return {'status':'scored' if n>50 else 'short_pair_score_provisional','frames':n,'embedding_count':n-5,'vshift':15,'valid_embedding_overlap_at_offset':max(0,n-5-abs(offset)),'confidence':conf,'offset_frames':offset,'confidence_pass':conf>3,'aligned_candidate':conf>3 and abs(offset)<=2,'robust_pass':bool(n>50 and conf>3 and abs(offset)<=2),'short_candidate':bool(n<=50 and conf>3 and abs(offset)<=2),'min_dist':float(dist.mean(0).min())}
  finally:
   import shutil
   wav.unlink(missing_ok=True);video.unlink(missing_ok=True);shutil.rmtree(self.tmp/'sync_tmp'/key,ignore_errors=True)
