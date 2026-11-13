"""Availability census, not training labels. Uses frozen CURRENT supplements only."""
import sqlite3,csv,json,hashlib,math,itertools,time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
SRC=ROOT/'experiments/p2_ltx_event_current_labels500h_v3_20261106'
META=ROOT/'experiments/p2_ltx_event_current_sources_v2_20261106'
OUT=Path(__file__).resolve().parent
MARGINS=(0.,.08,.16,.32)
CORE={'head_nod','head_shake','listener_vocalization'}
CLASSES=tuple(sorted(CORE|{'brow_lowering','brows_raised'}))

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def union(spans):
 out=[]
 for a,b in sorted(spans):
  if out and a<=out[-1][1]+1e-9:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def intersect(a,b):return union([[max(x,s),min(y,t)] for x,y in a for s,t in b if min(y,t)>max(x,s)])
def length(s):return sum(b-a for a,b in s)

def intervals(spans,duration):
 out=[]
 for a,b in spans:
  if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (a,b)) or a< -1e-6 or b>duration+.001001 or b<=a:raise ValueError('Invalid current interval')
  a,b=max(0.,a),min(duration,b)
  if b>a:out.append([a,b])
 return out

def current_events(row):
 p=row['production_source'];n=row['nod_source'];duration=row['duration_s'];dropped=Counter()
 if not p or not n or n['sample_id']!=row['sample_id'] or p['detector_sha256']!=n['detector_sha256']:raise ValueError('Missing or mismatched current detector')
 spans={'head_nod':intervals([[e['start'],e['end']] for e in n['events']],duration)}
 for c in ('head_shake','brow_lowering','brows_raised'):spans[c]=intervals(p['classes'][c]['spans'],duration)
 # Gate only CURRENT vocal cues. Historical audio positives never enter this census.
 raw=intervals(p['vocal_spans'],duration)
 spans['listener_vocalization']=intersect(raw,row['classes']['listener_vocalization']['spans'])
 dropped['vocal_seconds_outside_current_support']=length(union(raw))-length(spans['listener_vocalization'])
 for c in CLASSES:
  if c=='listener_vocalization':continue
  coverage=p['nod_coverage'] if c=='head_nod' else p['classes'][c]['coverage']
  unknown=[] if c=='head_nod' else p['classes'][c]['unknown']
  good=[]
  for a,b in spans[c]:
   if length(intersect([[a,b]],coverage))<b-a-1e-6 or length(intersect([[a,b]],unknown))>1e-8:dropped['visual_intervals_without_full_observational_coverage']+=1
   else:good.append([a,b])
  spans[c]=good
 # Connected intervals are support bouts, never claimed to be physical action counts.
 return {c:union(v) for c,v in spans.items()},dropped

def chunk(bounds):
 db=sqlite3.connect((SRC/'labels.sqlite').as_uri()+'?mode=ro',uri=True)
 counts=Counter();class_counts=Counter();event_counts=Counter();metrics=defaultdict(Counter);pairs=defaultdict(Counter);strict=[]
 for ordinal,sid,value in db.execute('SELECT ordinal,sid,value FROM labels WHERE ordinal>=? AND ordinal<? ORDER BY ordinal',bounds):
  row=json.loads(value)
  if row['sample_id']!=sid:raise ValueError('ID mismatch')
  events,drops=current_events(row);counts.update(drops);counts['samples']+=1
  present=[c for c in CLASSES if events[c]]
  class_counts.update(present);event_counts.update({c:len(events[c]) for c in present})
  counts['with_any_current_event']+=bool(present)
  counts['with_two_or_more_extended_classes']+=len(present)>=2
  counts['with_two_or_more_core_classes']+=len(set(present)&CORE)>=2
  flags=set()
  for a,b in itertools.combinations(present,2):
   aa,bb=events[a],events[b];core=a in CORE and b in CORE
   counts['co_present_class_pairs']+=1
   counts['class_pairs_with_overlapping_support']+=bool(intersect(aa,bb))
   counts['class_pairs_with_repeated_bouts']+=len(aa)>1 or len(bb)>1
   for first,second in ((a,b),(b,a)):
    x,y=events[first],events[second];gap=min(v[0] for v in y)-max(v[1] for v in x)
    for margin in MARGINS:
     for mode,valid in [('any_instance',any(s-v>=margin-1e-8 for u,v in x for s,t in y)),('class_separated',gap>=margin-1e-8),('single_bout',len(x)==len(y)==1 and gap>=margin-1e-8)]:
      if not valid:continue
      for scope in ('extended','core') if core else ('extended',):
       key=f'{scope}/{mode}/gap_{margin:.2f}'
       metrics[key]['relations']+=1;pairs[key][first+' -> '+second]+=1;flags.add(key)
    if len(x)==len(y)==1 and gap>=.32-1e-8:
     strict.append(dict(ordinal=ordinal,sample_id=sid,before=first,after=second,before_start=x[0][0],before_end=x[0][1],after_start=y[0][0],after_end=y[0][1],gap_s=gap,core_pair=core))
  for key in flags:metrics[key]['samples']+=1;metrics[key]['clip_seconds']+=row['duration_s']
 db.close()
 return dict(counts=counts,class_counts=class_counts,event_counts=event_counts,metrics=dict(metrics),pairs=dict(pairs),strict=strict)

def main():
 started=time.time()
 if (OUT/'COMPLETE.json').exists():raise FileExistsError('Completed audit already exists')
 m=json.loads((SRC/'manifest.json').read_text())
 if (SRC/'INVALIDATED.json').exists() or sha(SRC/'labels.sqlite')!=m['database_sha256']:raise ValueError('Invalidated or changed source')
 meta=list(csv.DictReader((META/'sample_list.csv').open()));ids={r['sample_id']:r for r in meta}
 if len(ids)!=len(meta) or len(meta)!=m['samples']:raise ValueError('Current population mismatch')
 counts=Counter();cc=Counter();ec=Counter();metrics=defaultdict(Counter);pairs=defaultdict(Counter);rows=[]
 with ProcessPoolExecutor(max_workers=8) as pool:
  for i,result in enumerate(pool.map(chunk,[(i,min(i+4096,len(meta))) for i in range(0,len(meta),4096)]),1):
   counts.update(result['counts']);cc.update(result['class_counts']);ec.update(result['event_counts']);rows+=result['strict']
   for k,v in result['metrics'].items():metrics[k].update(v)
   for k,v in result['pairs'].items():pairs[k].update(v)
   if i%10==0:print(json.dumps(dict(chunks=i,samples=counts['samples'],elapsed_s=round(time.time()-started,1))),flush=True)
 if counts['samples']!=len(meta):raise ValueError('Incomplete population')
 for k,v in metrics.items():v['hours']=v['clip_seconds']/3600;v['percent']=v['samples']/len(meta)*100
 fields=['ordinal','sample_id','split','duration_s','before','after','before_start','before_end','after_start','after_end','gap_s','core_pair']
 with (OUT/'strict_candidates.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:
   if meta[r['ordinal']]['sample_id']!=r['sample_id']:raise ValueError('Source order mismatch')
   w.writerow(dict(r,split=ids[r['sample_id']]['split'],duration_s=ids[r['sample_id']]['duration_s']))
 # Independent CSV readback: relation arithmetic, unique tuples, split and coverage.
 seen=set();sids=set();core_ids=set();types=Counter();splitids=defaultdict(set)
 for r in csv.DictReader((OUT/'strict_candidates.csv').open()):
  sid=r['sample_id'];key=(sid,r['before'],r['after'])
  if key in seen or r['before']==r['after']:raise ValueError('Duplicate or same-class relation')
  seen.add(key);sids.add(sid);splitids[r['split']].add(sid)
  a,b,c,d=(float(r[k]) for k in ('before_start','before_end','after_start','after_end'))
  if not 0<=a<b<=c<d<=float(r['duration_s'])+1e-6 or c-b<.32-1e-8 or abs(c-b-float(r['gap_s']))>1e-7:raise ValueError('Invalid exported relation')
  if r['core_pair']=='True':core_ids.add(sid)
  types[r['before']+' -> '+r['after']]+=1
 if len(sids)!=metrics['extended/single_bout/gap_0.32']['samples'] or len(seen)!=metrics['extended/single_bout/gap_0.32']['relations'] or len(core_ids)!=metrics['core/single_bout/gap_0.32']['samples']:raise ValueError('Export counts mismatch')
 sample_db=sqlite3.connect((SRC/'labels.sqlite').as_uri()+'?mode=ro',uri=True);src_db=sqlite3.connect((META/'sources.sqlite').as_uri()+'?mode=ro',uri=True)
 review=[]
 # Two examples per available pair type, selected by SHA to avoid prefix selection.
 buckets=defaultdict(list)
 for r in rows:buckets[(r['before'],r['after'])].append(r)
 for typ,rs in sorted(buckets.items()):
  for r in sorted(rs,key=lambda x:hashlib.sha256(x['sample_id'].encode()).hexdigest())[:2]:
   sid=r['sample_id'];target=json.loads(sample_db.execute('SELECT value FROM labels WHERE sid=?',(sid,)).fetchone()[0]);item=json.loads(src_db.execute('SELECT value FROM samples WHERE sid=?',(sid,)).fetchone()[0]);n=target['nod_source'];p=target['production_source']
   if sha(n['caption_file'])!=n['caption_file_sha256'] or sha(p['detector_file'])!=p['detector_sha256']:raise ValueError('Changed sampled evidence')
   cap=json.loads(Path(n['caption_file']).read_text());det=json.loads(Path(p['detector_file']).read_text())
   if cap['sample_id']!=sid or cap['source']!=item['source'] or cap['caption']!=item['caption'] or det['source']!=item['source']:raise ValueError('Sampled source mismatch')
   review.append(dict(relation=r,caption=item['caption'],input_evidence=cap['input_evidence'],detector_sha256=p['detector_sha256']))
 (OUT/'REVIEW_SAMPLES.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 result=dict(complete=True,population=len(meta),population_hours=sum(float(x['duration_s'])/3600 for x in meta),source_manifest=str(SRC/'manifest.json'),source_manifest_sha256=sha(SRC/'manifest.json'),database_sha256=m['database_sha256'],code_sha256=sha(__file__),counts=dict(counts),class_clip_counts=dict(cc),class_support_bouts=dict(ec),metrics={k:dict(v) for k,v in metrics.items()},pair_distributions={k:dict(v) for k,v in pairs.items()},strict_split_sample_counts={k:len(v) for k,v in splitids.items()},csv_sha256=sha(OUT/'strict_candidates.csv'),review_samples=len(review),caveats=['Current qualified detector snapshots only; no historical VL intervals and no caption keyword labels.','Single bout means one connected support interval, not one verified physical action.','Order here means end-before-start; overlaps/interleaving are excluded in strict modes.','Margins are sensitivity analyses, not estimates of detector accuracy.','Candidate availability is not validated event truth or final training admission.','Clip hours measure sample duration, not event or relation duration.'],elapsed_s=time.time()-started)
 (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,summary_sha256=sha(OUT/'SUMMARY.json'),review_sha256=sha(OUT/'REVIEW_SAMPLES.json'),csv_sha256=result['csv_sha256']),indent=2)+'\n')
 print(json.dumps(dict(counts=result['counts'],class_clip_counts=dict(cc),metrics=result['metrics'],strict_pairs=dict(types),review_samples=len(review),elapsed_s=result['elapsed_s']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
