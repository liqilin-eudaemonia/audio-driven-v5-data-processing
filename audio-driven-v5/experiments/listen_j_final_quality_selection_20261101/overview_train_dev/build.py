"""Read-only train+dev union overview. Original split fields are not modified."""
from pathlib import Path
import collections,csv,hashlib,json,math,sqlite3,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_listen_quality_selection import distributions,quantiles,write_csv,FACE,DURATION,RICHNESS,THRESHOLDS,case,label
OUT=Path(__file__).resolve().parent;WORK=OUT.parent
start=time.time();db=sqlite3.connect(f'file:{WORK}/selection.sqlite?mode=ro',uri=True)
db.execute('PRAGMA temp_store=MEMORY');db.execute('PRAGMA cache_size=-262144')
db.execute("CREATE TEMP VIEW samples AS SELECT * FROM main.samples WHERE split IN ('train','dev')")
db.execute('CREATE TEMP VIEW behaviors AS SELECT b.* FROM main.behaviors b JOIN samples s ON b.sid=s.sid')
n,h=db.execute('SELECT COUNT(*),SUM(duration)/3600 FROM samples').fetchone()
assert n==1095039
original=json.loads((WORK/'overview/overview.json').read_text())
assert math.isclose(h,sum(x['hours'] for x in original['split'] if x['split'] in ('train','dev')),abs_tol=1e-5)
r=dict(scope=['train','dev'],original_splits_unchanged=True,clips=n,hours=h,mean_duration_s=h*3600/n,
       split=[dict(split=s,clips=c,hours=t) for s,c,t in db.execute('SELECT split,COUNT(*),SUM(duration)/3600 FROM samples GROUP BY split')],
       source_pairs=db.execute('SELECT COUNT(DISTINCT pair) FROM samples').fetchone()[0],
       stat_status=dict(db.execute('SELECT status,COUNT(*) FROM samples GROUP BY status')))
r['quantiles']={k:quantiles(db,k) for k in ('duration','face_p10','face_p50','types')}
dists=[]
for metric,edges in [('duration',DURATION),('face_p10',FACE),('face_p50',FACE),('types',RICHNESS)]:dists+=distributions(db,metric,edges,n,h)
r['distributions']=dists;write_csv(OUT/'distributions.csv',dists)
thresholds=[]
for metric in ('face_p10','face_p50'):
 for minimum in THRESHOLDS:
  count,hours=db.execute(f'SELECT COUNT(*),COALESCE(SUM(duration)/3600,0) FROM samples WHERE {metric}>=?',(minimum,)).fetchone()
  thresholds.append(dict(metric=metric,minimum_px=minimum,clips=count,hours=hours,hours_pct=100*hours/h))
r['face_thresholds']=thresholds;write_csv(OUT/'face_thresholds.csv',thresholds)
capacity=[]
for minimum in (160,224,256,320,384):
 for types in (0,2,3,4,5):
  count,hours=db.execute('SELECT COUNT(*),COALESCE(SUM(duration)/3600,0) FROM samples WHERE face_p10>=? AND types>=?',(minimum,types)).fetchone()
  capacity.append(dict(face_p10_min_px=minimum,adopted_types_min=types,clips=count,hours=hours))
r['pool_capacity']=capacity;write_csv(OUT/'pool_capacity.csv',capacity)
print(json.dumps({'stage':'distributions','seconds':time.time()-start}),flush=True)
fields=['type','clips','source_pairs','hours','literal_occurrences','derived_occurrences','known_occurrence_clips','unknown_occurrence_clips','state_groups','pending_clips']
q='''SELECT type,SUM(counted),COUNT(DISTINCT CASE WHEN counted=1 THEN pair END),SUM(duration)/3600,
 SUM(CASE WHEN unit='occurrence' AND basis='literal' THEN quantity END),
 SUM(CASE WHEN unit='occurrence' AND basis='derived_from_distinct_mentions' THEN quantity END),
 SUM(CASE WHEN unit='occurrence' AND quantity IS NOT NULL THEN 1 ELSE 0 END),
 SUM(CASE WHEN unit='occurrence' AND quantity IS NULL THEN 1 ELSE 0 END),SUM(state),SUM(pending)
 FROM behaviors GROUP BY type'''
adopted=[];pending=[];original_categories={x['type']:x for x in csv.DictReader((WORK/'overview/all_categories.csv').open(encoding='utf-8-sig'))}
for values in db.execute(q):
 c=dict(zip(fields,values));c['modality']=c['type'].split(':',1)[0];c['clip_pct']=100*c['clips']/n;c['hours_pct']=100*c['hours']/h
 if c['clips']:
  old=original_categories[c['type']]
  assert c['clips']==int(old['train_clips'])+int(old['dev_clips'])
  assert math.isclose(c['hours'],float(old['train_hours'])+float(old['dev_hours']),abs_tol=1e-5)
  c['display_name']=old['display_name'];adopted.append(c)
 else:pending.append(c)
adopted.sort(key=lambda c:(-c['hours'],c['type']));write_csv(OUT/'all_categories.csv',adopted);write_csv(OUT/'pending_only_categories.csv',pending)
r['category_counts']=dict(adopted=len(adopted),by_modality=dict(collections.Counter(c['modality'] for c in adopted)),pending_only=len(pending))
r['top_categories']=adopted[:30]
r['category_rarity']=[dict(range=name,names=sum(lo<=c['clips']<hi for c in adopted)) for name,lo,hi in [('1',1,2),('2-9',2,10),('10-99',10,100),('100-999',100,1000),('1000-9999',1000,10000),('>=10000',10000,math.inf)]]
by_type={c['type']:c for c in adopted};cross=[]
for metric in ('face_p10','face_p50'):
 totals=collections.Counter()
 for key,bin_id,count,hours in db.execute(f'''SELECT b.type,{case('s.'+metric,FACE)} AS bin,COUNT(*),SUM(s.duration)/3600
  FROM behaviors b JOIN samples s ON b.sid=s.sid WHERE b.counted=1 GROUP BY b.type,bin'''):
  cross.append(dict(type=key,metric=metric,bin='unknown' if bin_id<0 else label(FACE[bin_id],FACE[bin_id+1]),clips=count,hours=hours));totals[key]+=count
 assert all(totals[k]==c['clips'] for k,c in by_type.items())
write_csv(OUT/'category_face_distribution.csv',cross)
assert sum(c['clips'] for c in adopted)==db.execute('SELECT SUM(types) FROM samples').fetchone()[0]
assert db.execute("SELECT COUNT(*) FROM samples WHERE split='test'").fetchone()[0]==0
assert not db.execute('SELECT pair FROM samples GROUP BY pair HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchall()
r['seconds']=time.time()-start;r['definitions']=original['definitions']
(OUT/'overview.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
audit=dict(passed=True,at=time.time(),test_excluded=True,splits_unchanged=True,clips=n,hours=h,
 categories_match_train_plus_dev_from_previous_overview=True,distributions_sum_to_scope=True,
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.csv','.json','.py') and p.name!='audit.json'})
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'complete':True,'clips':n,'hours':h,'categories':r['category_counts'],'seconds':r['seconds']}),flush=True)
