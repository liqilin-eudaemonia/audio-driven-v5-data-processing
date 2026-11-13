"""Freeze existing unified tuning items and independent-source holdout candidates."""
from pathlib import Path
import csv,json,sqlite3,hashlib,random,time,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/'experiments/p2_ltx_event_current_sources_v2_20261106'
UNIFIED=ROOT/'experiments/seamless_quick200_latest_20261105'

def pair_of(sid):
 if '_local_' not in sid:raise ValueError('Unknown sample naming: '+sid)
 return sid.rsplit('_local_',1)[0]

def freeze():
 manifest=read(SOURCE/'manifest.json');db=SOURCE/'sources.sqlite'
 if sha(db)!=manifest['database_sha256']:raise ValueError('Training source database changed')
 with (SOURCE/'sample_list.csv').open()as f:training=list(csv.DictReader(f))
 ids={x['sample_id']for x in training};pairs={pair_of(s)for s in ids}
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)as conn:
  rows=conn.execute("SELECT DISTINCT json_extract(value,'$.source.group_id'),json_extract(value,'$.source.pair') FROM samples").fetchall()
 groups={g for g,p in rows};source_pairs={p for g,p in rows}
 if None in groups or None in source_pairs or source_pairs!=pairs:raise ValueError('Missing source lineage')
 selected=[json.loads(s)for s in (UNIFIED/'selection.jsonl').read_text().splitlines()if s.strip()]
 if len(selected)!=200 or len({r['id']for r in selected})!=200:raise ValueError('Expected exact unified200')
 overlaps={kind:[]for kind in ('sample_id','pair','group')}
 for r in selected:
  if r['id']in ids:overlaps['sample_id'].append(r['id'])
  if r['pair']in pairs:overlaps['pair'].append(r['id'])
  if not r.get('group_id'):raise ValueError('Missing tuning group')
  if r['group_id']in groups:overlaps['group'].append(r['id'])
 if any(overlaps.values()):raise ValueError('Tuning/train overlap: '+json.dumps(overlaps))
 # Frozen, approximate stratification by caption hints and audio duration group.
 # Hints are automatic candidate annotations, not independent human truth.
 def stratum(r):
  h=r.get('selection_hints',{});visual=any(h.get(k)for k in ('head_nod','head_shake','smile','laugh','frown','surprise'))
  return (r.get('audio_group','unknown'),visual)
 buckets={}
 for r in sorted(selected,key=lambda r:r['id']):buckets.setdefault(stratum(r),[]).append(r)
 rng=random.Random(137)
 for values in buckets.values():rng.shuffle(values)
 tuning=[]
 while len(tuning)<64:
  for key in sorted(buckets):
   if buckets[key]and len(tuning)<64:tuning.append(buckets[key].pop())
 tuning_ids={r['id']for r in selected};tuning_pairs={r['pair']for r in selected};tuning_groups={r['group_id']for r in selected}
 candidates=[];seen=set();excluded_ids=ids|tuning_ids;excluded_pairs=pairs|tuning_pairs;excluded_groups=groups|tuning_groups
 with(UNIFIED/'candidates.latest.jsonl').open()as f:
  for line in f:
   r=json.loads(line)
   if r.get('split')!='test' or not r.get('group_id'):continue
   if r['sample_id']in excluded_ids or r['pair']in excluded_pairs or r['group_id']in excluded_groups or r['group_id']in seen:continue
   candidates.append(r);seen.add(r['group_id'])
 rng.shuffle(candidates);holdout=candidates[:128]
 for name,value in [('tuning200.json',selected),('tuning64.json',tuning),('holdout_candidates128.json',holdout)]:atomic_json(WORK/name,value,immutable=True)
 result=dict(success=True,at=time.time(),training_samples=len(ids),training_pairs=len(pairs),training_groups=len(groups),tuning_samples=200,screening_samples=64,train_tuning_overlaps=overlaps,holdout_candidate_samples=len(holdout),holdout_candidate_scope='Disjoint sample/pair/component from final500h and current unified200; preprocessing and prior global evaluation-exposure audit pending; not yet final-test acceptance',roles=dict(tuning='Existing unified200 repurposed for hyperparameter selection',holdout='Do not use to rank candidates during this search'),sources={str(SOURCE/'sample_list.csv'):sha(SOURCE/'sample_list.csv'),str(db):manifest['database_sha256'],str(UNIFIED/'selection.jsonl'):sha(UNIFIED/'selection.jsonl'),str(UNIFIED/'candidates.latest.jsonl'):sha(UNIFIED/'candidates.latest.jsonl')},outputs={name:sha(WORK/name)for name in ('tuning200.json','tuning64.json','holdout_candidates128.json')})
 atomic_json(WORK/'VALIDATION_MEMBERSHIP_ACCEPTED.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':freeze()
