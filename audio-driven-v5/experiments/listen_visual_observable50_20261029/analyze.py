"""Paired audit and descriptive diagnostics; no media-level accuracy claims."""
import collections, hashlib, json, re, sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from run import ADDITION
sys.path.insert(0,str(R.parents[1]/'src'))
from listen_benchmark.full_raw import views
from listen_benchmark.simple_timeline import validate_timeline
markers=re.compile(r'\b(indicat\w*|suggest\w*|impl[yi]\w*|signall?ing|convey\w*|engage\w*|attenti\w*|interest\w*|amuse\w*|enjoy\w*|thoughtful\w*|contemplat\w*|reflect\w*|process\w*|agree\w*|understand\w*|acknowledg\w*|empath\w*|bored\w*|confus\w*|distress\w*|sadness|likely|perhaps|possibly|intent\w*|motivat\w*|as if|react\w*|respon[ds]\w*|surpris\w*|curios\w*|skeptic\w*)\b',re.I)
explicit=re.compile(r'\b(indicat\w*|suggest\w*|impl[yi]\w*|signall?ing|convey\w*|as if|perhaps|possibly)\b',re.I)
acts={k:re.compile(v,re.I) for k,v in {'nod':r'\bnod\w*','shake':r'\bshak\w*','smile':r'\bsmil\w*','laugh':r'\b(laugh\w*|chuckl\w*|giggl\w*)','hand':r'\b(hand\w*|arm\w*)','brow':r'\b(eyebrow\w*|brow\w*|frown\w*)','eye':r'\b(eye\w*|blink\w*)','tilt':r'\btilt\w*','mouth':r'\b(mouth|lip\w*)'}.items()}
ARMS=['baseline','observable']; stats={a:collections.Counter() for a in ARMS}; full={a:collections.Counter() for a in ARMS}
clips={a:set() for a in ARMS}; flagclips={a:set() for a in ARMS}; actclips={a:collections.defaultdict(set) for a in ARMS}
paired=[];examples=[];errors=[];texts_out={a:[] for a in ARMS};files=sorted((R/'results').glob('*.json'))
samples=json.loads((R/'samples.json').read_text());expected={r['sample_id'] for r in samples};seen=set()
for p in files:
 row=json.loads(p.read_text());sid=row['sample_id'];assert sid in expected and sid not in seen;seen.add(sid);by={}
 for rec in row['records']:
  a=rec['arm'];s=stats[a];s['requests']+=1;s['valid_requests']+=rec['status']=='valid';s['truncated']+=rec['finish_reason']!='stop';s['completion_tokens']+=rec['completion_tokens']
  assert a not in by.setdefault(rec['view'],{});by[rec['view']][a]=rec
  if rec['status']!='valid':errors.append(dict(sample_id=sid,view=rec['view'],arm=a,error=rec.get('error')));continue
  es=rec['parsed']['events'];validate_timeline(rec['parsed'],rec['end']-rec['start']);assert json.loads(rec['raw'])==rec['parsed'];assert all(e['modality']=='visual' for e in es)
  s['valid_events']+=len(es);s['nonempty_requests']+=bool(es)
  if es:clips[a].add(sid)
  counter=collections.Counter();counter['valid_requests']=1;counter['valid_events']=len(es);counter['nonempty_requests']=bool(es)
  exact=[]
  for e in es:
   text=e['behavior'];hits=markers.findall(text);counter['description_chars']+=len(text);counter['flagged_descriptions']+=bool(hits);counter['explicit_marker_descriptions']+=bool(explicit.search(text));counter['shorter_than_0_2s']+=e['end']-e['start']<.2-1e-7
   if hits:flagclips[a].add(sid)
   for k,pat in acts.items():
    found=bool(pat.search(text));counter['action_'+k]+=found
    if found:actclips[a][k].add(sid)
   exact.append(text.casefold().strip());texts_out[a].append(dict(sample_id=sid,view=rec['view'],start=e['start']+rec['start'],end=e['end']+rec['start'],behavior=text,flags=hits))
  counter['exact_repeated_descriptions_within_request']=len(exact)-len(set(exact))
  counter['requests_with_flags']=counter['flagged_descriptions']>0
  counter['requests_with_explicit_markers']=counter['explicit_marker_descriptions']>0
  for k,v in counter.items():
   if k not in ('valid_requests','valid_events','nonempty_requests'):s[k]+=v
  if rec['view']=='full':full[a].update(counter)
 for name,start,end in views(row['duration_s']):
  arms=by[name];assert set(arms)==set(ARMS)
  b,o=arms['baseline'],arms['observable'];assert b['prompt']+'\n'+ADDITION==o['prompt'];assert b['media']==o['media'];assert b['start']==o['start']==start and b['end']==o['end']==end
  texts={a:[e['behavior'] for e in v.get('parsed',{}).get('events',[])] for a,v in arms.items()}
  flags={a:sum(bool(markers.search(t)) for t in ts) for a,ts in texts.items()};actsets={a:{k for k,pat in acts.items() if any(pat.search(t) for t in ts)} for a,ts in texts.items()}
  paired.append(dict(sample_id=sid,view=name,statuses={a:r['status'] for a,r in arms.items()},flagged=flags,baseline_only_action_words=sorted(actsets['baseline']-actsets['observable']),observable_only_action_words=sorted(actsets['observable']-actsets['baseline'])))
  if flags['observable'] or (flags['baseline'] and name=='full'):examples.append(dict(sample_id=sid,view=name,texts=texts))
 assert set(by)=={v[0] for v in views(row['duration_s'])}
report=dict(completed_samples=len(files),planned_samples=50,all_samples_complete=seen==expected,stats={a:dict(s,clips_with_events=len(clips[a]),clips_with_flags=len(flagclips[a]),clips_with_action_mentions={k:len(ids) for k,ids in actclips[a].items()}) for a,s in stats.items()},full_view_stats={a:dict(c) for a,c in full.items()},errors=errors,lexical_flags_are_not_semantic_truth=True,actions_are_mentions_not_verified_recall=True,human_boundary_truth_available=False,paired=paired,examples=examples)
(R/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for a,es in texts_out.items():(R/f'descriptions_{a}.json').write_text(json.dumps(es,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['paired','examples']},ensure_ascii=False,indent=2))
