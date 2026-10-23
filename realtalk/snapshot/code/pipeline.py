"""Full RealTalk pair manifests and native-resolution evidence; never exports crops.
Frame-exhaustive ledger. Approximate ASR boundaries remain explicitly reviewable.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,math,re,time,subprocess,concurrent.futures,traceback,hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SRC=Path('/public/0_DATA/2_Avatar/realtalk');FPS=25;VERSION='realtalk-pairs-10s-max11-v1'
REF=Path('/private/liqi_workspace/realtalk_inspection/full_video_v2_sentence/review/samples.json')
POLICY={'version':VERSION,'target_seconds':10,'maximum_seconds':11,'min_seconds':None,'scene_threshold':.12,'black_pixel_threshold':.12,'black_picture_fraction':.90,'scene_guard_frames':0,'bridge_unknown_max_frames':20,'face_resolution_filter':None,'crop_resolution_filter':None,'audio_quality_filter':None,'asr_sentence_end_is_candidate_not_ground_truth':True,'incomplete_sentence_is_not_rejection':True,'all_frames_accounted':True,'native_crop_no_upscale_in_statistics':True}
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');tmp.replace(path)
def spans(values):
 if len(values)==0:return []
 out=[]
 for i,k in enumerate(values):
  if not out or out[-1][2]!=k:out.append([i,i+1,k])
  else:out[-1][1]=i+1
 return out

def annotation_arrays(ann,n):
 role=np.full(n,-1,np.int8);boxes=np.full((n,2,4),np.nan,np.float32);scores=np.full((n,2),np.nan,np.float32)
 for key,f in ann.items():
  i=int(key)
  if not 0<=i<n:continue
  speaker=f.get('current_speaker');role[i]=0 if speaker=='p0' else 1 if speaker=='p1' else -1
  for p in range(2):
   q=f.get('people',{}).get('p'+str(p),{});box=q.get('bbox')
   if box is not None and len(box)==4:boxes[i,p]=box
   if q.get('score') is not None:scores[i,p]=q['score']
 return role,boxes,scores

def scan_source(item,ann):
 vid=item['video_id'];cache=ROOT/'scans'/(vid+'.json');energyfile=ROOT/'scans'/(vid+'_energy.npy')
 if cache.exists() and energyfile.exists():return json.loads(cache.read_text()),np.load(energyfile)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',item['source']]))
 vs=next(s for s in probe['streams'] if s['codec_type']=='video');fps=float(Fraction(vs['avg_frame_rate']))
 if abs(fps-FPS)>1e-5:raise ValueError('unsupported fps '+str(fps))
 n=int(vs.get('nb_frames') or max(map(int,ann))+1);width,height=int(vs['width']),int(vs['height'])
 rawrole,boxes,scores=annotation_arrays(ann,n);geom=set()
 for p in range(2):
  a=boxes[:-1,p];b=boxes[1:,p];size=np.maximum(np.maximum(a[:,2]-a[:,0],a[:,3]-a[:,1]),1);motion=np.linalg.norm((a[:,:2]+a[:,2:]-b[:,:2]-b[:,2:])/2,axis=1)/size;ratio=np.maximum(np.maximum(b[:,2]-b[:,0],b[:,3]-b[:,1]),1)/size
  geom.update((np.where((motion>.45)|(ratio>1.35)|(ratio<1/1.35))[0]+1).tolist())
 log=ROOT/'logs'/(vid+'_scan.log')
 cmd=['ffmpeg','-v','info','-nostdin','-threads',os.environ.get('REALTALK_DECODE_THREADS','1'),'-filter_threads','1','-i',item['source'],'-map','0:v:0','-an','-vf',"scale=320:180,blackdetect=d=0.04:pix_th=0.12:pic_th=0.90,select='gt(scene,0.12)',showinfo",'-f','null','/dev/null','-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
 with log.open('wb') as f:raw=subprocess.check_output(cmd,stderr=f)
 wave=np.frombuffer(raw,np.float32);nn=min(n,len(wave)//640);energy=np.zeros(n,np.float32)
 if nn:energy[:nn]=np.sqrt(np.mean(wave[:nn*640].reshape(-1,640).astype(np.float64)**2,axis=1))
 txt=log.read_text(errors='replace');cuts=sorted(set(round(float(x)*FPS) for x in re.findall(r'pts_time:([\d.]+)',txt)));black=[(round(float(a)*FPS),round(float(b)*FPS)) for a,b in re.findall(r'black_start:([\d.]+) black_end:([\d.]+)',txt)]
 edges=sorted({0,n}|{x for x in cuts+list(geom)+[x for ab in black for x in ab] if 0<=x<=n})
 scan=dict(video_id=vid,frames=n,width=width,height=height,fps=FPS,scene_cuts=cuts,geometry_cuts=sorted(geom),black=black,edges=edges,annotation_frames=len(ann),annotation_last_frame=max(map(int,ann)),audio_complete_frames=nn,codec=vs['codec_name'])
 np.save(energyfile,energy);save(cache,scan);return scan,energy

def shot_crops(item,scan,boxes):
 cache=ROOT/'scans'/(item['video_id']+'_crops.json')
 if cache.exists():return json.loads(cache.read_text())
 width,height=scan['width'],scan['height'];shots=list(zip(scan['edges'],scan['edges'][1:]));sampleids={};wanted=[]
 for j,(lo,hi) in enumerate(shots):
  if hi-lo<15:continue
  if not np.isfinite(boxes[lo:hi]).any():continue
  ids=sorted(set(min(hi-1,max(lo,math.ceil(t))) for t in np.linspace(lo+6.25,hi-6.25,3)));sampleids[j]=ids;wanted.extend(ids)
 wanted=sorted(set(wanted));measures={}
 if wanted:
  expr='+'.join(f'eq(n\\,{i})' for i in wanted);cmd=['ffmpeg','-v','error','-nostdin','-threads',os.environ.get('REALTALK_DECODE_THREADS','1'),'-filter_threads','1','-i',item['source'],'-an','-vf',f"fps=25:start_time=0,select='{expr}'",'-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1'];proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE);size=width*height*3
  for idx in wanted:
   buf=proc.stdout.read(size)
   if len(buf)!=size:raise ValueError(f'frame sampling incomplete at {idx}: {len(buf)}')
   im=np.frombuffer(buf,np.uint8).reshape(height,width,3).astype(np.int16);diff=np.mean(np.abs(im[:,1:]-im[:,:-1]),axis=2);measures[idx]=(diff.mean(0),(diff>15).mean(0))
  err=proc.stderr.read();assert proc.wait()==0,err[-500:]
 result=[]
 for j,(lo,hi) in enumerate(shots):
  seams=[];ids=sampleids.get(j,[])
  if ids:
   score=np.median([measures[i][0] for i in ids],axis=0);coverage=np.median([measures[i][1] for i in ids],axis=0);good=(score>18)&(coverage>.45)
   for a,b,k in spans(good):
    if k:
     x=int(a+np.argmax(score[a:b])+1)
     if width*.078<x<width*.922:seams.append(dict(x=x,strength=float(score[x-1]),coverage=float(coverage[x-1])))
   clean=[]
   for s in sorted(seams,key=lambda z:-z['strength']):
    if not any(abs(s['x']-q['x'])<8*width/1280 for q in clean):clean.append(s)
   seams=sorted(clean,key=lambda z:z['x'])
   if not seams:
    a=max(1,round(width*.4375));b=min(width-1,round(width*.5625));peaks=[int(a+np.argmax(measures[i][0][a-1:b-1])) for i in ids];x=int(np.median(peaks));margin=max(5,round(40*width/1280));near=np.r_[score[max(0,x-margin):max(0,x-5)],score[x+5:min(width-1,x+margin)]]
    if len(near) and max(peaks)-min(peaks)<=max(4,4*width/1280) and score[x-1]>3 and score[x-1]>3*max(float(np.median(near)),.5):seams=[dict(x=x,strength=float(score[x-1]),method='persistent_low_contrast_seam')]
  crops={}
  for p in range(2):
   bs=boxes[lo:hi,p];valid=np.isfinite(bs).all(1)&(bs[:,2]>bs[:,0])&(bs[:,3]>bs[:,1]);bs=bs[valid];crop=None
   if len(bs) and ids:
    med=np.median(bs,axis=0);bw=float(med[2]-med[0]);bh=float(med[3]-med[1]);left=max([s['x'] for s in seams if s['x']<med[0]-6]+[0]);right=min([s['x'] for s in seams if s['x']>med[2]+6]+[width]);pl=math.ceil((left+(4 if left else 0))/2)*2;pr=math.floor((right-(4 if right<width else 0))/2)*2
    if width*.15625<=pr-pl<=width*.546875:
     size=min(math.ceil(max(bw,bh)*2.35/2)*2,pr-pl,height//2*2);x=int(np.clip((med[0]+med[2])/2-size/2,pl,pr-size))//2*2;y=int(np.clip((med[1]+med[3])/2-size*.6,0,height-size))//2*2
     crop=dict(x=x,y=y,width=size,height=size,native_edge=size,panel=[pl,pr],method='shot_median_and_measured_panel',status='geometry_candidate')
   crops['p'+str(p)]=crop
  result.append(dict(number=j+1,start_frame=lo,end_frame=hi,crops=crops,seams=seams))
 save(cache,result);return result

def prepare_asr(item,n):
 d=json.loads(Path(item['asr']).read_text());candidates={}
 for s in d.get('segments',[]):
  try:a=float(s['start']);b=float(s['end'])
  except (KeyError,ValueError,TypeError):continue
  if not 0<=a<b<=n/FPS+1:continue
  if re.search(r'[.!?]["\u201d\u2019\']?\s*$',s.get('text','').strip()):
   f=math.ceil(b*FPS-1e-7)
   candidates[f]=dict(reason='source_asr_sentence_candidate',cost=1.,reference_text=s['text'],asr_segment_id=s.get('id'),review=True)
 return candidates

def partition(a,b,asr,energy):
 if b-a<=275:return [a,b],{}
 choices={f:dict(v) for f,v in asr.items() if a<f<b};rms=energy[a:b];threshold=max(.001,float(np.percentile(rms,90))*.12)
 for lo,hi,k in spans(rms<threshold):
  if k and hi-lo>=3:
   f=a+(lo+hi)//2
   if a<f<b and f not in choices:choices[f]=dict(reason='acoustic_pause_candidate',cost=2.,pause_frames=hi-lo,review=True)
 # Every frame is available as a high-cost fallback; no data is removed.
 N=b-a;dp=np.full(N+1,np.inf);prev=np.zeros(N+1,np.int32);dp[0]=0
 for j in range(1,N+1):
  lo=max(0,j-275);idx=np.arange(lo,j);dur=(j-idx)/FPS;pen=0 if j==N else choices.get(a+j,{'cost':40})['cost'];values=dp[idx]+2+pen+.03*(dur-10)**2+np.where(dur<1,5,0);k=int(np.argmin(values));dp[j]=values[k];prev[j]=idx[k]
 bounds=[N]
 while bounds[-1]>0:bounds.append(int(prev[bounds[-1]]))
 bounds=[a+x for x in reversed(bounds)];selected={f:choices.get(f,dict(reason='frame_fallback_needs_alignment',cost=40,review=True)) for f in bounds[1:-1]};return bounds,selected

def bbox_stats(bs,width,height):
 valid=np.isfinite(bs).all(1)&(bs[:,2]>bs[:,0])&(bs[:,3]>bs[:,1]);raw=bs[valid];out_of_bounds=int(np.sum((raw[:,0]<0)|(raw[:,1]<0)|(raw[:,2]>width)|(raw[:,3]>height)))
 clipped=raw.copy()
 if len(clipped):clipped[:,[0,2]]=np.clip(clipped[:,[0,2]],0,width);clipped[:,[1,3]]=np.clip(clipped[:,[1,3]],0,height)
 w=clipped[:,2]-clipped[:,0] if len(clipped) else np.array([]);h=clipped[:,3]-clipped[:,1] if len(clipped) else np.array([]);good=(w>0)&(h>0);w=w[good];h=h[good];short=np.minimum(w,h)
 def summary(x):return dict(min=round(float(np.min(x)),3),p10=round(float(np.percentile(x,10)),3),median=round(float(np.median(x)),3),p90=round(float(np.percentile(x,90)),3),max=round(float(np.max(x)),3)) if len(x) else None
 # Exact 1px bins, weighted by 1/FPS seconds. Underflow/out-of-frame are separate.
 hist=dict(zip(*np.unique(np.floor(short).astype(int),return_counts=True))) if len(short) else {}
 return dict(valid_frames=int(len(short)),missing_or_invalid_frames=int(len(bs)-len(short)),out_of_bounds_frames=out_of_bounds,width=summary(w),height=summary(h),short_side=summary(short)),{str(int(k)):int(v) for k,v in hist.items()}

def process(item):
 starttime=time.time();vid=item['video_id'];dest=ROOT/'manifests'/(vid+'.json')
 if dest.exists():return dict(video_id=vid,cached=True)
 ann=json.loads(Path(item['annotation']).read_text());scan,energy=scan_source(item,ann);n=scan['frames'];rawrole,boxes,scores=annotation_arrays(ann,n);role=rawrole.copy();edges=scan['edges'];bridged=np.zeros(n,bool)
 for a,b,k in spans(role):
  if k==-1 and a>0 and b<n and b-a<=20 and role[a-1]==role[b] and role[b]>=0 and not any(a<=e<=b for e in edges):
   other=1-int(role[b])
   if np.all(np.nan_to_num(scores[a:b,other],nan=0)<=.2):role[a:b]=role[b];bridged[a:b]=True
 black=np.zeros(n,bool)
 for a,b in scan['black']:black[max(0,a):min(n,b)]=True
 crops=shot_crops(item,scan,boxes);asr=prepare_asr(item,n);ledger=[];pairs=[];facehist={};reference=None
 if vid=='-1wCGAxqT_4':reference=json.loads(REF.read_text());assert n==reference['frames']
 if reference:
  rawsegments=[]
  for r in reference['segments']:
   rr=dict(start_frame=r['start_frame'],end_frame=r['end_frame'],shot=r['shot'],kind=r['kind'])
   if r['kind']=='pair':rr.update(speaker=r['speaker'],listener=r['listener'],reference=r)
   rawsegments.append(rr)
 else:
  rawsegments=[]
  for shot,(lo,hi) in enumerate(zip(edges,edges[1:]),1):
   states=np.where(black[lo:hi],-2,role[lo:hi])
   for a,b,k in spans(states):
    a+=lo;b+=lo
    if k<0:rawsegments.append(dict(start_frame=a,end_frame=b,shot=shot,kind='black' if k==-2 else 'role_unknown'));continue
    bounds,boundaries=partition(a,b,asr,energy)
    for x,y in zip(bounds,bounds[1:]):rawsegments.append(dict(start_frame=x,end_frame=y,shot=shot,kind='pair',speaker='p'+str(k),listener='p'+str(1-k),continuous_interval=[a,b],end_boundary=boundaries.get(y,dict(reason='hard_interval_end',review=False)),start_boundary=boundaries.get(x,dict(reason='hard_interval_start',review=False))))
 for s in rawsegments:
  a,b=s['start_frame'],s['end_frame'];s.update(start=a/FPS,end=b/FPS,duration=(b-a)/FPS)
  if s['kind']=='pair':
   p=int(s['listener'][1]);stats,hist=bbox_stats(boxes[a:b,p],scan['width'],scan['height']);
   for k,v in hist.items():facehist[k]=facehist.get(k,0)+v
   crop=crops[s['shot']-1]['crops'][s['listener']]
   if 'reference' in s:
    r=s.pop('reference');c=r['crop'];crop=None if c is None else dict(x=c['x'],y=c['y'],width=c['size'],height=c['size'],native_edge=c['size'],panel=c['panel'],method='approved_pilot_reference',status='geometry_candidate');s['end_boundary']=r.get('end_boundary',dict(reason='approved_reference_boundary',review=True));s['start_boundary']=r.get('start_boundary',dict(reason='approved_reference_boundary',review=True));s['reference_pair_id']=r['pair_id']
   flags=[]
   if stats['missing_or_invalid_frames']:flags.append('face_annotation_missing_or_invalid')
   if stats['out_of_bounds_frames']:flags.append('face_box_clipped_to_source_for_statistics')
   if crop is None:flags.append('pre_crop_unresolved')
   if bridged[a:b].any():flags.append('short_role_gap_bridged')
   positive=float(np.mean(np.nan_to_num(scores[a:b,p],nan=0)>.2))
   if positive>.08:flags.append('listener_active_score_conflict')
   if s['duration']<1:flags.append('short_pair_retained')
   if s['end_boundary'].get('review') or s['start_boundary'].get('review'):flags.append('boundary_review')
   s.update(pair_id=f'{vid}__{len(pairs)+1:05d}',video_id=vid,crop=crop,face=stats,flags=flags,listener_active_fraction=round(positive,5),training_admitted=False)
   pairs.append(s)
  else:s['reason']='black_detected' if s['kind']=='black' else 'speaker_role_unresolved_not_discarded'
  ledger.append(s)
 assert ledger[0]['start_frame']==0 and ledger[-1]['end_frame']==n and all(a['end_frame']==b['start_frame'] for a,b in zip(ledger,ledger[1:]));assert all(0<r['duration']<=11 for r in pairs)
 # Every raw known-role, nonblack frame must remain a candidate.
 covered=np.zeros(n,bool)
 for r in pairs:covered[r['start_frame']:r['end_frame']]=True
 assert np.all(covered[(rawrole>=0)&~black])
 stats={k:sum(r['end_frame']-r['start_frame'] for r in ledger if r['kind']==k) for k in ['pair','role_unknown','black']};assert sum(stats.values())==n
 result=dict(version=VERSION,video_id=vid,source=item,width=scan['width'],height=scan['height'],fps=FPS,frames=n,duration=n/FPS,pairs=pairs,ledger=ledger,frame_totals=stats,listener_face_short_side_frame_histogram=facehist,crop_unresolved_pair_frames=sum(r['end_frame']-r['start_frame'] for r in pairs if r['crop'] is None),face_valid_frames=sum(r['face']['valid_frames'] for r in pairs),boundary_methods=dict(__import__('collections').Counter(r['end_boundary']['reason'] for r in pairs)),reference_manifest_sha256=hashlib.sha256(REF.read_bytes()).hexdigest() if reference else None,checks={'exhaustive_frame_ledger':True,'known_role_frames_retained':True,'max11_seconds':True},elapsed_seconds=round(time.time()-starttime,2))
 save(dest,result);return dict(video_id=vid,pairs=len(pairs),frames=n,seconds=round(time.time()-starttime,1))

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=24);parser.add_argument('--limit',type=int);parser.add_argument('--video');args=parser.parse_args();items=json.loads((ROOT/'index.json').read_text());save(ROOT/'policy.json',POLICY)
 if args.video:items=[r for r in items if r['video_id']==args.video]
 if args.limit:items=items[:args.limit]
 errors=[];done=0;started=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(process,r):r for r in items}
  for future in concurrent.futures.as_completed(futures):
   r=futures[future]
   try:out=future.result();done+=1;print(json.dumps(dict(done=done,total=len(items),**out)),flush=True)
   except Exception as e:
    error=dict(video_id=r['video_id'],error=str(e),traceback=traceback.format_exc());errors.append(error);save(ROOT/'logs'/(r['video_id']+'_error.json'),error);print('ERROR',r['video_id'],str(e),flush=True)
   save(ROOT/'progress.json',dict(total=len(items),completed=done,errors=errors,elapsed_seconds=round(time.time()-started,1),running=done+len(errors)<len(items)))
 print('COMPLETE',done,'errors',len(errors),flush=True)
if __name__=='__main__':main()
