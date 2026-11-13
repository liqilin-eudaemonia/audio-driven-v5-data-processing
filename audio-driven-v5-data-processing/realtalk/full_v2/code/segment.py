from pathlib import Path
import json,sys,subprocess,re,hashlib,time,os
import numpy as np,soundfile as sf
from boundary_core import *
ROOT=Path(__file__).resolve().parents[1];R=Path(os.environ['REALTALK_VIDEO_WORK']);BASE=ROOT.parent/'full_20261023';VID=os.environ['REALTALK_VIDEO_ID']
def save(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x));tmp.replace(p)
def crop_for(bs,p,w,h):
 own=bs[:,p];other=bs[:,1-p];own=np.asarray([q for q in own if valid_box(q)]);other=np.asarray([q for q in other if valid_box(q)])
 if not len(own):return None
 q=np.median(own,axis=0);left,top,right,bottom=0.,0.,float(w),float(h)
 if len(other):
  z=np.median(other,axis=0);cx,cy=(q[:2]+q[2:])/2;ox,oy=(z[:2]+z[2:])/2;gx=abs(cx-ox)-(q[2]-q[0]+z[2]-z[0])/2;gy=abs(cy-oy)-(q[3]-q[1]+z[3]-z[1])/2
  if max(gx,gy)<0:return None
  if gx>=gy:
   if cx<ox:right=(q[2]+z[0])/2
   else:left=(z[2]+q[0])/2
  else:
   if cy<oy:bottom=(q[3]+z[1])/2
   else:top=(z[3]+q[1])/2
 edge=int(min(np.ceil(max(q[2]-q[0],q[3]-q[1])*2.35),right-left,bottom-top))//2*2
 if edge<16:return None
 x=int(np.clip((q[0]+q[2])/2-edge/2,left,right-edge));y=int(np.clip((q[1]+q[3])/2-edge*.6,top,bottom-edge));coverage=float(np.mean((own[:,0]>=x)&(own[:,1]>=y)&(own[:,2]<=x+edge)&(own[:,3]<=y+edge)))
 return {'x':x,'y':y,'width':edge,'height':edge,'native_edge':edge,'face_box_coverage':coverage,'method':'per_pair_identity_partition_native_crop','review':coverage<.95 or not len(other)}
def main():
 m=json.loads((BASE/'manifests'/f'{VID}.json').read_text());meta=m['source'];inputs={'source':meta,'archive_stat':[Path(meta['archive']).stat().st_size,Path(meta['archive']).stat().st_mtime_ns],'annotation_sha256':hashlib.sha256(Path(meta['annotation']).read_bytes()).hexdigest(),'word_sha256':hashlib.sha256((R/'words.json').read_bytes()).hexdigest(),'audio_sha256':hashlib.sha256((R/'source_audio.wav').read_bytes()).hexdigest(),'video_sha256':hashlib.sha256((R/'source_video.mp4').read_bytes()).hexdigest()};code_files=[Path(__file__).parent/'boundary_core.py',Path(__file__)];inputs['processing_code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files};sig=fingerprint(code_files,inputs)
 if (R/'manifest.json').exists() and json.loads((R/'manifest.json').read_text()).get('run_fingerprint')==sig:print('CACHE_MATCH',flush=True);return
 # Actual normalized decoded frame count is authoritative, not AVI header nb_frames.
 raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1','-filter_threads','1','-i',str(R/'source_video.mp4'),'-vf','scale=160:90,format=gray','-an','-f','rawvideo','pipe:1']);gray=np.frombuffer(raw,np.uint8).reshape(-1,90,160);n=len(gray);w,h=m['width'],m['height'];black=np.mean(gray<31,axis=(1,2))>=.9
 ann=json.loads(Path(meta['annotation']).read_text());role=np.full(n,-1,int);boxes=np.full((n,2,4),np.nan);scores=np.full((n,2),np.nan)
 for key,f in ann.items():
  i=int(key)
  if i>=n:continue
  who=f.get('current_speaker');role[i]=int(who[1]) if who in ['p0','p1'] else -1
  for p in [0,1]:
   q=f.get('people',{}).get(f'p{p}',{});b=q.get('bbox')
   if b is not None and len(b)==4:boxes[i,p]=b
   if q.get('score') is not None:scores[i,p]=q['score']
 cmd=['ffmpeg','-nostdin','-v','info','-threads','1','-filter_threads','1','-i',str(R/'source_video.mp4'),'-an','-vf',"scale=320:180,select='gt(scene,0.12)',showinfo",'-f','null','-'];scan=subprocess.run(cmd,capture_output=True,text=True,check=True);(R/'logs/normalized_scene_scan.log').write_text(scan.stderr);scenes=sorted({round(float(x)*25) for x in re.findall(r'pts_time:([\d.]+)',scan.stderr) if 0<round(float(x)*25)<n})
 geom=set()
 for p in [0,1]:
  for i in range(1,n):
   a,b=boxes[i-1,p],boxes[i,p]
   if not(valid_box(a) and valid_box(b)):continue
   scale=max(a[2]-a[0],a[3]-a[1],1);motion=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/scale;ratio=max(b[2]-b[0],b[3]-b[1])/scale
   if motion>.45 or ratio>1.35 or ratio<1/1.35:geom.add(i)
 hard,geom_review=reconcile_geometry(boxes,geom,scenes,gray,w,h);blackedges={x for a,b,k in spans(black) if k for x in [a,b]};edges=sorted({0,n}|set(hard)|blackedges)
 wave,sr=sf.read(R/'source_audio.wav');assert sr==16000;af=min(n,len(wave)//640);energy=np.full(n,np.nan);energy[:af]=np.sqrt(np.mean(wave[:af*640].reshape(-1,640)**2,axis=1));states,events,smooth=refine_roles(role,scores,edges,black,af);words=json.loads((R/'words.json').read_text())['words'];ledger=[];pairs=[]
 cut_reasons={e:'scene_cut' if e in scenes else 'confirmed_layout_change' for e in hard};cut_reasons.update({e:'black_boundary' for e in blackedges});cut_reasons.update({0:'source_start',n:'source_end',af:'actual_audio_end'})
 for ev in events:
  if ev['type']=='formal_takeover_candidate':cut_reasons[ev['start_frame']]='formal_takeover_onset_candidate'
  if ev['type']=='takeover_overlap_retained_for_review':cut_reasons[ev['start_frame']]='formal_takeover_overlap_onset';cut_reasons[ev['end_frame']]='overlap_end_candidate'
 for shot,(lo,hi) in enumerate(zip(edges,edges[1:]),1):
  for a,b,k in spans(states[lo:hi]):
   a+=lo;b+=lo
   if k<0:
    kind={-1:'role_unknown',-2:'black',-3:'overlap_transition',-4:'audio_unavailable'}[k];ledger.append({'kind':kind,'start_frame':a,'end_frame':b,'start':a/25,'end':b/25,'duration':(b-a)/25,'shot':shot,'review':kind!='black'});continue
   cuts,internal=partition(a,b,words,energy)
   for x,y in zip(cuts,cuts[1:]):
    crop=crop_for(boxes[x:y],1-k,w,h);ev=[e for e in events if e['start_frame']<y and e['end_frame']>x];pair={'kind':'pair','pair_id':f'{VID}__v2_{len(pairs)+1:04d}','video_id':VID,'start_frame':x,'end_frame':y,'start':x/25,'end':y/25,'duration':(y-x)/25,'shot':shot,'continuous_interval':[a,b],'speaker':f'p{k}','listener':f'p{1-k}','crop':crop,'role_evidence_events':ev,'start_boundary':internal.get(x,{'reason':cut_reasons.get(x,'role_interval_start'),'review':x not in scenes}),'end_boundary':internal.get(y,{'reason':cut_reasons.get(y,'role_interval_end'),'review':y not in scenes}),'words':[q for q in words if q['start']<y/25 and q['end']>x/25],'training_admitted':False,'role_semantics_verified':False,'review':True};pairs.append(pair);ledger.append(pair)
 assert ledger[0]['start_frame']==0 and ledger[-1]['end_frame']==n and all(a['end_frame']==b['start_frame'] for a,b in zip(ledger,ledger[1:]));assert sum(x['end_frame']-x['start_frame'] for x in ledger)==n
 for p in pairs:
  assert 0<p['end_frame']-p['start_frame']<=275 and p['end_frame']<=af;assert not black[p['start_frame']:p['end_frame']].any();assert not any(p['start_frame']<e<p['end_frame'] for e in hard)
 summary={'video_id':VID,'source_seconds':n/25,'source_frames':n,'audio_complete_frames':af,'audio_actual_samples':len(wave),'pair_count':len(pairs),'candidate_seconds':sum(p['duration'] for p in pairs),'old_pair_count':len(m['pairs']),'old_candidate_seconds':sum(p['duration'] for p in m['pairs']),'max_pair_seconds':max((p['duration'] for p in pairs),default=0),'one_frame_pairs':sum(p['end_frame']-p['start_frame']==1 for p in pairs),'old_one_frame_pairs':sum(p['end_frame']-p['start_frame']==1 for p in m['pairs']),'scene_cuts':len(scenes),'geometry_proposals':len(geom),'confirmed_extra_geometry_cuts':len(set(hard)-set(scenes)),'event_counts':{t:sum(e['type']==t for e in events) for t in sorted({e['type'] for e in events})},'ledger_seconds':{t:sum(x['duration'] for x in ledger if x['kind']==t) for t in sorted({x['kind'] for x in ledger})},'all_frames_classified':True,'semantics':'heuristic boundary candidates; no training admission; unknown/overlap preserved'}
 save(R/'manifest.json',{'version':POLICY['version'],'run_fingerprint':sig,'inputs':inputs,'video_id':VID,'width':w,'height':h,'fps':25,'frames':n,'source':meta,'policy':POLICY,'pairs':pairs,'ledger':ledger,'events':events,'scene_edges':edges,'geometry_review':geom_review,'summary':summary});save(R/'summary.json',summary);save(R/'policy.json',POLICY);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
