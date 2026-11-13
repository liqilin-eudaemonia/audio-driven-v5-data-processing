"""Versioned boundary helpers; no I/O and no mutation of legacy candidates."""
import numpy as np,math,re,hashlib,json
FPS=25
POLICY={'version':'realtalk-boundary-v2-pilot','target_seconds':10,'max_frames':275,'score_positive':.2,'dominance_margin':.5,'formal_support_frames':20,'short_return_frames':20,'unknown_bridge_frames':20,'onset_search_frames':25,'geometry_persistence_frames':3,'geometry_scene_snap_frames':2,'black_gray_threshold':31,'black_fraction':.9,'no_missing_scores_as_silence':True,'role_thresholds':'heuristic_candidates_require_review','scene_threshold':.12}
def spans(a):
 out=[]
 for i,x in enumerate(a):
  x=int(x)
  if out and out[-1][2]==x:out[-1][1]=i+1
  else:out.append([i,i+1,x])
 return out

def valid_box(b):return np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]
def smooth_scores(x):
 y=np.full_like(x,np.nan,dtype=float)
 for i in range(len(x)):
  q=x[max(0,i-2):i+3]
  for p in [0,1]:
   v=q[:,p];v=v[np.isfinite(v)]
   if np.isfinite(x[i,p]) and len(v)>=min(3,len(q)):y[i,p]=np.median(v)
 return y

def refine_roles(raw,scores,edges,black,audio_frames):
 role=raw.copy();events=[];s=smooth_scores(scores)
 for lo,hi in zip(edges,edges[1:]):
  if hi<=lo:continue
  # Both actual score observations are required for dominance, never impute missing data.
  good=np.isfinite(s[lo:hi]).all(1);dom=np.full(hi-lo,-1,int)
  for p in [0,1]:dom[good&(s[lo:hi,p]>.2)&(s[lo:hi,p]-s[lo:hi,1-p]>.5)]=p
  for a,b,p in spans(dom):
   a+=lo;b+=lo
   if p>=0 and b-a>=20 and np.any(role[a:b]!=p):
    events.append({'start_frame':a,'end_frame':b,'type':'sustained_speaking_evidence','speaker':p,'review':True});role[a:b]=p
  # Brief opposite turns returning to the prior active speaker are preserved as backchannel candidates.
  for a,b,p in spans(role[lo:hi]):
   a+=lo;b+=lo
   if p<0 or b-a>=20 or a==lo or b==hi:continue
   parent=int(role[a-1])
   if parent<0 or parent==p or role[b]!=parent:continue
   q=np.r_[s[max(lo,a-5):a,parent],s[b:min(hi,b+5),parent]]
   if len(q) and np.isfinite(q).all() and np.mean(q>.2)>=.6 and np.isfinite(s[a:b]).all():
    role[a:b]=parent;events.append({'start_frame':a,'end_frame':b,'type':'short_return_backchannel_candidate','speaker':parent,'responder':p,'review':True})
  # Bridge unknown gaps only with explicit negative opposite activity and observed scores.
  for a,b,p in spans(role[lo:hi]):
   a+=lo;b+=lo
   if p!=-1 or b-a>20 or a==lo or b==hi:continue
   parent=int(role[a-1]);q=s[a:b]
   if parent>=0 and role[b]==parent and np.isfinite(q).all() and np.all(q[:,1-parent]<=.2):
    role[a:b]=parent;events.append({'start_frame':a,'end_frame':b,'type':'observed_inactive_gap_bridge','speaker':parent,'review':True})
  # Backdate a supported takeover to its local active onset; keep sustained overlap separately.
  runs=spans(role[lo:hi])
  for k in range(1,len(runs)):
   a,b,p=runs[k];a+=lo;b+=lo;prev=runs[k-1][2]
   if p<0 or prev<0 or p==prev or b-a<20:continue
   onset=a
   while onset>max(lo,a-25) and np.isfinite(s[onset-1]).all() and s[onset-1,p]>.2:onset-=1
   onset=max(onset,lo+runs[k-1][0])
   if onset<a:role[onset:a]=p
   end=onset
   while end<min(b,onset+50) and np.isfinite(s[end]).all() and s[end,p]>.2 and s[end,prev]>.2:end+=1
   if end-onset>=20:
    role[onset:end]=-3;events.append({'start_frame':onset,'end_frame':end,'type':'takeover_overlap_retained_for_review','previous_speaker':prev,'next_speaker':p,'review':True})
   events.append({'start_frame':onset,'end_frame':b,'type':'formal_takeover_candidate','previous_speaker':prev,'speaker':p,'original_transition_frame':a,'review':True})
 role[np.asarray(black,bool)]=-2
 role[audio_frames:]=np.where(np.asarray(black[audio_frames:],bool),-2,-4)
 return role,events,s

def reconcile_geometry(boxes,candidates,scenes,gray,w,h):
 accepted=[];review=[];n=len(boxes)
 for t in sorted(set(candidates)):
  near=[x for x in scenes if abs(x-t)<=2]
  if near:review.append({'frame':t,'decision':'snapped_to_scene','scene':min(near,key=lambda x:abs(x-t))});continue
  if t<3 or t+3>n:review.append({'frame':t,'decision':'insufficient_temporal_support'});continue
  persistent=[]
  for p in [0,1]:
   pre=boxes[t-3:t,p];post=boxes[t:t+3,p]
   if not all(valid_box(q) for q in np.r_[pre,post]):continue
   x=np.median(pre,axis=0);y=np.median(post,axis=0);scale=max(x[2]-x[0],x[3]-x[1],1);dist=np.linalg.norm((x[:2]+x[2:]-y[:2]-y[2:])/2)/scale;ratio=max(y[2]-y[0],y[3]-y[1])/scale;stable=max(np.linalg.norm((q[:2]+q[2:]-y[:2]-y[2:])/2)/max(y[2]-y[0],y[3]-y[1],1) for q in post)<.15
   if stable and (dist>.45 or ratio>1.35 or ratio<1/1.35):persistent.append(p)
  before=np.median(gray[t-3:t],axis=0);after=np.median(gray[t:t+3],axis=0);change=float(np.mean(np.abs(before-after))/255)
  ok=bool(persistent) and change>.10
  review.append({'frame':t,'decision':'confirmed_geometry_and_visual_change' if ok else 'unconfirmed_geometry_change','persistent_identities':persistent,'visual_mean_difference':change})
  if ok:accepted.append(t)
 # Adjacent geometry estimates are one event; visual scene boundaries remain untouched.
 clustered=[]
 for t in accepted:
  if clustered and t-clustered[-1]<=2:continue
  clustered.append(t)
 return sorted(set(scenes)|set(clustered)),review

def partition(a,b,words,energy):
 if b-a<=275:return [a,b],{}
 candidates={};occupied=np.zeros(b-a,bool)
 for w in words:
  lo=max(a,math.floor(w['start']*25));hi=min(b,math.ceil(w['end']*25));f=math.ceil(w['end']*25-1e-7)
  if hi>lo:occupied[lo-a:hi-a]=True
  if a<f<b:
   sentence=bool(re.search(r'[.!?]["\u201d\u2019\']?$',w['text']));v={'reason':'word_sentence_end' if sentence else 'word_end','cost':1 if sentence else 6,'text':w['text'],'word_time':[w['start'],w['end']],'review':True}
   if f not in candidates or v['cost']<candidates[f]['cost']:candidates[f]=v
 rms=energy[a:b];threshold=max(.001,float(np.percentile(rms,90))*.12)
 for lo,hi,k in spans((rms<threshold)&~occupied):
  if k and hi-lo>=3:
   f=a+(lo+hi)//2
   if a<f<b and f not in candidates:candidates[f]={'reason':'word_gap_acoustic_pause','cost':2,'review':True}
 N=b-a;dp=np.full(N+1,np.inf);prev=np.zeros(N+1,int);dp[0]=0
 for j in range(1,N+1):
  idx=np.arange(max(0,j-275),j);dur=(j-idx)/25;pen=0 if j==N else candidates.get(a+j,{'cost':100 if occupied[j-1] else 40})['cost'];v=dp[idx]+2+pen+.03*(dur-10)**2+np.where(dur<1,5,0);k=int(np.argmin(v));dp[j]=v[k];prev[j]=idx[k]
 ends=[N]
 while ends[-1]:ends.append(int(prev[ends[-1]]))
 bounds=[a+x for x in ends[::-1]];return bounds,{f:candidates.get(f,{'reason':'frame_fallback_needs_review','review':True}) for f in bounds[1:-1]}
def fingerprint(code_files,inputs,policy=POLICY):
 return hashlib.sha256(json.dumps({'code':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},'inputs':inputs,'policy':policy},sort_keys=True).encode()).hexdigest()
