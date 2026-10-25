"""One GPU worker, one streaming decode per source, atomic pair checkpoints."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,collections,traceback,time,argparse,shutil
import numpy as np,soundfile as sf
from engine import Engine,R,BASE,VERSION,atomic,ff,context_interval,recover_crop,valid_box
from calibrate import rms,db,scale
from gate import make_gain

def arrays(m):
 ann=json.loads(Path(m['source']['annotation']).read_text());n=m['frames'];boxes=np.full((n,2,4),np.nan,np.float32);scores=np.full((n,2),np.nan,np.float32);role=np.full(n,-1,np.int8)
 for k,v in ann.items():
  i=int(k)
  if i>=n:continue
  who=v.get('current_speaker');role[i]=int(who[1]) if who in ('p0','p1') else -1
  for p in range(2):
   person=v.get('people',{}).get(f'p{p}',{});b=person.get('bbox')
   if b is not None and len(b)==4:boxes[i,p]=b
   if person.get('score') is not None:scores[i,p]=person['score']
 role[:]=-1
 for p in m['pairs']:role[p['start_frame']:p['end_frame']]=int(p['speaker'][1])
 return boxes,scores,role

def calibration_jobs(m,boxes,scores,role,maxframe=None):
 out=[]
 for person in (0,1):
  candidates=[]
  for p in m['pairs']:
   a,b=p['start_frame'],p['end_frame']
   if p['speaker']!=f'p{person}' or b-a<125 or (maxframe is not None and b>maxframe):continue
   lo=a;hi=a+125;good=(role[lo:hi]==person)&(scores[lo:hi,person]>=1)&(scores[lo:hi,1-person]<=0)&np.isfinite(boxes[lo:hi]).all(axis=(1,2));quality=float(np.mean(good))
   if quality>=.8:candidates.append((quality,p,lo,hi))
  for quality,p,lo,hi in sorted(candidates,key=lambda x:(-x[0],x[2])):
   if any(max(lo,c['lo'])<min(hi,c['hi']) or abs(lo-c['lo'])<250 for c in out if c['speaker']==person):continue
   out.append({'kind':'calibration','id':f"cal_p{person}_{sum(c['speaker']==person for c in out)}",'lo':lo,'hi':hi,'shot':p['shot'],'speaker':person,'quality':quality})
   if sum(c['speaker']==person for c in out)>=2:break
 return out

def trusted_cache(path,manifest_hash,code_hash):
 if not path.exists():return None
 try:
  r=json.loads(path.read_text())
  if r['manifest_sha256']!=manifest_hash or r['code_sha256']!=code_hash:return None
  if r.get('phase')=='retryable_error' or any(r.get(k,{}).get('status')=='scoring_error' for k in ['raw_sync','optimized_sync']):return None
  if r.get('calibration_file') and hashlib.sha256(Path(r['calibration_file']).read_bytes()).hexdigest()!=r['calibration_sha256']:return None
  if r.get('source_signature'):
   st=Path(r['source_signature']['path']).stat();assert st.st_size==r['source_signature']['size'] and st.st_mtime_ns==r['source_signature']['mtime_ns']
  for key in ['raw_audio','optimized_audio']:
   if key in r:
    f=Path(r[key]);assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==r[key+'_sha256'];a,sr=sf.read(f);assert sr==16000 and len(a)==r['frames']*640
  if r.get('phase')=='complete':return r
  if r.get('phase')=='raw_complete' and Path(r['face_video']).exists():return r
 except Exception:return None
 return None

def process_video(task,engine,out,code_hash,limit_pairs=None):
 started=time.time();m=json.loads(Path(task['manifest']).read_text());assert hashlib.sha256(Path(task['manifest']).read_bytes()).hexdigest()==task['manifest_sha256'];vid=m['video_id'];pairs=m['pairs'][:limit_pairs] if limit_pairs else m['pairs'];resultdir=out/'results'/vid;resultdir.mkdir(parents=True,exist_ok=True);audiodir=out/'audio'/vid;audiodir.mkdir(parents=True,exist_ok=True);work=engine.tmp/vid;work.mkdir(parents=True,exist_ok=True);status=out/'video_status'/f'{vid}.json';cached={p['pair_id']:trusted_cache(resultdir/f"{p['pair_id']}.json",task['manifest_sha256'],code_hash) for p in pairs};done={k:r for k,r in cached.items() if r and r.get('phase')=='complete'}
 if len(done)==len(pairs):atomic(status,{'video_id':vid,'status':'complete','pairs':len(pairs),'completed':len(done),'manifest_sha256':task['manifest_sha256'],'code_sha256':code_hash,'reused':True});return
 boxes,scores,role=arrays(m);edges=m['scene_edges'];engine.det_cache={};raw_results={k:r for k,r in cached.items() if r};caljobs=calibration_jobs(m,boxes,scores,role,max(p['end_frame'] for p in pairs) if limit_pairs else None);jobs=caljobs.copy();cal=[]
 for p in pairs:
  if cached[p['pair_id']]:continue
  lo,hi=context_interval(p,edges);jobs.append({'kind':'pair','id':p['pair_id'],'lo':lo,'hi':hi,'shot':p['shot'],'pair':p})
 jobs.sort(key=lambda j:(j['hi'],j['lo'],j['id']));retain_frames=max([300]+[j['hi']-j['lo']+1 for j in jobs]);source_stat=Path(m['source']['archive']).stat();source_signature={'path':m['source']['archive'],'size':source_stat.st_size,'mtime_ns':source_stat.st_mtime_ns};atomic(status,{'video_id':vid,'status':'decoding','pairs':len(pairs),'completed':len(done),'started_at':started,'worker_pid':os.getpid()})
 raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1','-i',m['source']['source'],'-vn','-af','aresample=16000:async=1:first_pts=0','-ac','1','-ar','16000','-f','f32le','pipe:1']);audio=np.frombuffer(raw,np.float32);maxend=max([j['hi'] for j in jobs],default=0);w,h=m['width'],m['height'];size=w*h*3;ring={};nextjob=0
 log=(out/'logs'/f'{vid}_decode.log').open('ab');proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','1','-filter_threads','1','-i',m['source']['source'],'-vf','fps=25:start_time=0','-frames:v',str(maxend),'-an','-pix_fmt','bgr24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=log)
 try:
  for index in range(maxend):
   data=proc.stdout.read(size)
   if len(data)!=size:raise RuntimeError(f'source_decode_incomplete:{index}/{maxend}')
   ring[index]=np.frombuffer(data,np.uint8).reshape(h,w,3)
   if index-retain_frames in ring:del ring[index-retain_frames]
   if index-retain_frames in engine.det_cache:del engine.det_cache[index-retain_frames]
   while nextjob<len(jobs) and jobs[nextjob]['hi']==index+1:
    j=jobs[nextjob];nextjob+=1;lo,hi=j['lo'],j['hi'];mix=audio[lo*640:hi*640]
    if j['kind']=='pair':
     p=j['pair'];a,b=p['start_frame'],p['end_frame'];key=p['pair_id'];rec={'version':VERSION,'code_sha256':code_hash,'model_fingerprints':engine.model_fingerprints,'source_signature':source_signature,'manifest_sha256':task['manifest_sha256'],'video_id':vid,'pair_id':key,'canonical_pair':p,'speaker':p['speaker'],'listener':p['listener'],'frames':b-a,'duration':(b-a)/25,'context_frames':[lo,hi],'context_only_for_separation_and_tracking':True,'training_admitted':False,'phase':'complete'}
    try:
     if len(mix)!=(hi-lo)*640:raise ValueError('source_audio_incomplete')
     persons=[f'p{j["speaker"]}',f'p{1-j["speaker"]}'] if j['kind']=='calibration' else [p['listener']];outputs={}
     for person in persons:
      track,evidence=engine.track(person,lo,hi,ring,boxes,edges,j['shot'],w,h);ctxface=work/(j['id']+'_'+person+'_ctx.avi');faces,visual=engine.faces(ring,lo,track,ctxface);z,padding=engine.separate(mix,visual);outputs[person]={'audio':z,'padding':padding}
      if j['kind']=='pair':
       sl=slice(a-lo,b-lo);targettrack=track[sl];other=boxes[a:b,1-int(person[1])];other=[q for q in other if valid_box(q,w,h)];crop=p['crop']
       if crop is None:crop=recover_crop(targettrack,np.median(other,axis=0) if other else None,w,h)
       if crop is None:raise ValueError('crop_unresolved:unable_to_isolate_identity')
       extra=[];visible=0;speaker_support=[]
       from engine import iou
       for frame in sorted(set([a,(a+b-1)//2,b-1])):
        detections=engine.detect(ring[frame],frame);own=targettrack[frame-a]
        if any(iou(own,q)>=.1 for q in detections):visible+=1
        speaker_box=boxes[frame,1-int(person[1])]
        if valid_box(speaker_box,w,h) and any(iou(speaker_box,q)>=.1 and iou(speaker_box,q)>iou(own,q)+.05 for q in detections):speaker_support.append(frame)
        for q in detections:
         cx,cy=(q[:2]+q[2:])/2
         if iou(own,q)<.1 and crop['x']<=cx<crop['x']+crop['width'] and crop['y']<=cy<crop['y']+crop['height']:extra.append(q)
       if not visible:raise ValueError('identity_unresolved:no_detection_inside_original_pair')
       if extra:
        repaired=recover_crop(targettrack,np.median(extra,axis=0),w,h)
        if repaired is None:raise ValueError('crop_conflict:multiple_detected_faces')
        for q in extra:
         cx,cy=(q[:2]+q[2:])/2
         if repaired['x']<=cx<repaired['x']+repaired['width'] and repaired['y']<=cy<repaired['y']+repaired['height']:raise ValueError('crop_conflict:cannot_isolate_target')
        crop=repaired
       centers=(targettrack[:,:2]+targettrack[:,2:])/2;inside=(centers[:,0]>=crop['x'])&(centers[:,0]<crop['x']+crop['width'])&(centers[:,1]>=crop['y'])&(centers[:,1]<crop['y']+crop['height'])
       if np.mean(inside)<.95:raise ValueError('crop_conflict:target_center_outside_original_crop')
       pairaudio=z[(a-lo)*640:(b-lo)*640];path=audiodir/(key+'_raw.wav');sf.write(path,pairaudio,16000,subtype='FLOAT');face=audiodir/(key+'_face.avi');engine.write_face(faces[sl],face);gain=min(1.,.95/max(float(np.max(np.abs(audio[a*640:b*640]))),float(np.max(np.abs(pairaudio))),1e-10));score=engine.score(face,pairaudio,key+'_raw',gain)
       rec.update(role_review_required=not bool(speaker_support),role_evidence={'speaker_detection_support_frames':speaker_support,'listener_detection_samples':visible,'note':'Roles come from approved v2 boundary refinement, not the old pair list. Detection confirms identity only, not semantic turn-taking. Missing opposite identity support requires review.'},effective_crop=crop,crop_recovered=crop!=p['crop'],identity=evidence,raw_audio=str(path),raw_audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),face_video=str(face),raw_sync=score,common_preview_gain=gain,inference_padding_frames=padding,phase='raw_complete',source_audio_samples=[a*640,b*640])
      ctxface.unlink(missing_ok=True)
     if j['kind']=='calibration':
      level=scale(mix);mr=rms(mix);ids=np.minimum(np.arange(len(mr))*160//640,hi-lo-1)+lo;s=j['speaker'];mask=(role[ids]==s)&(scores[ids,s]>=1)&(scores[ids,1-s]<=0)&(mr>=level*.1);vals={p:{'kind':'positive' if p==f'p{s}' else 'negative','values':db(rms(v['audio'])[mask]/level).tolist()} for p,v in outputs.items()};cal.append({'id':j['id'],'start_frame':lo,'end_frame':hi,'people':vals,'annotation_verified_not_human_verified':True})
    except Exception as exc:
     if j['kind']=='pair':
      expected=isinstance(exc,ValueError) and str(exc).startswith(('identity_unresolved:','track_unresolved:','crop_unresolved:','crop_conflict:'))
      rec.update(status='needs_review' if expected else 'retryable_error',reason=str(exc),phase='complete' if expected else 'retryable_error',optimized_sync={'status':'not_attempted_without_raw','robust_pass':False});atomic(resultdir/f'{key}.json',rec);raw_results[key]=rec
     else:cal.append({'id':j['id'],'start_frame':lo,'end_frame':hi,'error':str(exc)})
     print('JOB_WARNING',vid,j['id'],str(exc),flush=True)
    else:
     if j['kind']=='pair':atomic(resultdir/f'{key}.json',rec);raw_results[key]=rec
    if nextjob%25==0:atomic(status,{'video_id':vid,'status':'separating','pairs':len(pairs),'raw_records':len(raw_results),'completed':len(done),'source_frame':index,'source_frames':maxend,'elapsed_seconds':time.time()-started})
  assert nextjob==len(jobs);assert proc.wait()==0
 finally:
  if proc.poll() is None:
   proc.stdout.close();proc.terminate()
   try:proc.wait(timeout=3)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()
 atomic(out/'calibration'/f'{vid}.json',{'video_id':vid,'records':cal,'code_sha256':code_hash,'manifest_sha256':task['manifest_sha256']})
 cal_live=out/'calibration'/f'{vid}.json';cal_digest=hashlib.sha256(cal_live.read_bytes()).hexdigest();cal_snapshot=out/'calibration'/f'{vid}_{cal_digest}.json'
 if not cal_snapshot.exists():shutil.copyfile(cal_live,cal_snapshot)
 for p in pairs:
  key=p['pair_id'];rec=raw_results[key]
  if rec['phase']!='raw_complete':continue
  lo,hi=rec['context_frames'];person=p['listener'];pools={'positive':[],'negative':[]};used=[]
  for c in cal:
   if 'error' in c or max(lo,c['start_frame'])<min(hi,c['end_frame']):continue
   vals=c['people'][person];pools[vals['kind']].extend(vals['values']);used.append(c['id'])
  a,sr=sf.read(rec['raw_audio']);mix=audio[p['start_frame']*640:p['end_frame']*640]
  if min(map(len,pools.values()))<50:
   rec['optimized_sync']={'status':'independent_calibration_insufficient','robust_pass':False,'positive_windows':len(pools['positive']),'negative_windows':len(pools['negative'])}
  else:
   wanted=float(np.quantile(pools['negative'],.95))+3;cap=float(np.quantile(pools['positive'],.01))-6;th=min(wanted,cap)+float(db(scale(mix)));g,_=make_gain(a,{'threshold_dbfs':th,'bridge_low_gap_ms':50,'pre_roll_ms':40,'post_roll_ms':60,'fade_ms':20});z=(a*g).astype(np.float32);f=audiodir/(key+'_optimized.wav');sf.write(f,z,16000,subtype='FLOAT');rec.update(optimized_audio=str(f),optimized_audio_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),optimization={'threshold_dbfs':th,'muted_seconds':float(np.sum(g==0)/16000),'calibration_ids':used,'positive_windows':len(pools['positive']),'negative_windows':len(pools['negative']),'exclude_pair_and_context_overlap':True});rec['optimized_sync']=engine.score(rec['face_video'],z,key+'_optimized',rec['common_preview_gain'])
  calfile=cal_snapshot;rec['calibration_file']=str(calfile);rec['calibration_sha256']=hashlib.sha256(calfile.read_bytes()).hexdigest();rec['phase']='retryable_error' if any(rec.get(k,{}).get('status')=='scoring_error' for k in ['raw_sync','optimized_sync']) else 'complete';rec['status']='processed' if rec['phase']=='complete' else 'retryable_error';atomic(resultdir/f'{key}.json',rec)
 failures=[r for r in raw_results.values() if r['phase']!='complete'];atomic(status,{'video_id':vid,'status':'error' if failures else 'complete','retryable_pairs':len(failures),'pairs':len(pairs),'completed':len(pairs)-len(failures),'elapsed_seconds':time.time()-started,'manifest_sha256':task['manifest_sha256'],'code_sha256':code_hash,'calibration_records':len(cal)});shutil.rmtree(work,ignore_errors=True);print('VIDEO_DONE',vid,len(pairs),round(time.time()-started,2),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=8);ap.add_argument('--root',default=str(R));ap.add_argument('--video',action='append');ap.add_argument('--limit-pairs',type=int);a=ap.parse_args();out=Path(a.root)
 for d in ['results','video_status','audio','logs','calibration']:(out/d).mkdir(parents=True,exist_ok=True)
 engine=Engine(f'/tmp/realtalk_full_sync_{os.getpid()}');codehash=hashlib.sha256(b''.join((R/'code'/n).read_bytes() for n in ['engine.py','worker.py','gate.py','calibrate.py'])+json.dumps(engine.model_fingerprints,sort_keys=True).encode()).hexdigest();tasks=json.loads((R/'queue.json').read_text());loads=[0]*a.shards;shards=[[] for _ in loads]
 for task in sorted(tasks,key=lambda x:-x['source_frames']):
  i=int(np.argmin(loads));shards[i].append(task);loads[i]+=task['source_frames']
 selected=[x for x in tasks if x['video_id'] in a.video] if a.video else shards[a.shard]
 for task in selected:
  try:process_video(task,engine,out,codehash,a.limit_pairs)
  except Exception as exc:atomic(out/'video_status'/f"{task['video_id']}.json",{'video_id':task['video_id'],'status':'error','error':str(exc),'traceback':traceback.format_exc(),'pairs':task['pairs'],'code_sha256':codehash});print('VIDEO_ERROR',task['video_id'],str(exc),flush=True)
 shutil.rmtree(engine.tmp,ignore_errors=True)
if __name__=='__main__':main()
