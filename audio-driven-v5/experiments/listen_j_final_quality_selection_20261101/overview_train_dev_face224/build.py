"""Read-only statistics of final train+dev candidates with native face p10>=224."""
import collections,csv,json,sqlite3,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parent
db=sqlite3.connect(f'file:{WORK}/selection.sqlite?mode=ro',uri=True)
db.execute('PRAGMA temp_store=MEMORY')
db.execute("CREATE TEMP TABLE chosen AS SELECT * FROM samples WHERE split IN ('train','dev') AND face_p10>=224")
db.execute('CREATE UNIQUE INDEX temp.chosen_id ON chosen(sid)')
n,secs=db.execute('SELECT COUNT(*),SUM(duration) FROM chosen').fetchone();assert n==267070
names={r['type']:r['display_name'] for r in csv.DictReader((WORK/'overview/all_categories.csv').open(encoding='utf-8-sig'))}
categories=[]
for typ,c,sec in db.execute('SELECT b.type,COUNT(*),SUM(s.duration) FROM behaviors b JOIN chosen s ON s.sid=b.sid WHERE b.counted=1 GROUP BY b.type ORDER BY COUNT(*) DESC'):
 categories.append(dict(type=typ,name=names.get(typ,typ),clips=c,clip_pct=100*c/n,coverage_hours=sec/3600))
with (OUT/'categories.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(categories[0]));w.writeheader();w.writerows(categories)
result=dict(clips=n,hours=secs/3600,split=[dict(split=s,clips=c,hours=h) for s,c,h in db.execute('SELECT split,COUNT(*),SUM(duration)/3600 FROM chosen GROUP BY split')],category_names=len(categories),category_names_by_modality=dict(collections.Counter(c['type'].split(':')[0] for c in categories)),categories=categories,types_per_clip=[dict(types=t,clips=c,hours=h) for t,c,h in db.execute('SELECT types,COUNT(*),SUM(duration)/3600 FROM chosen GROUP BY types')])
a=collections.Counter();flags=collections.Counter();rows=db.execute('SELECT sid,file,offset FROM chosen ORDER BY file,offset');prev=None;f=None;start=time.time()
for i,(sid,path,offset) in enumerate(rows,1):
 if path!=prev:
  if f:f.close()
  f=open(path,'rb');prev=path
 f.seek(offset);r=json.loads(f.readline());assert r['sample_id']==sid
 au=r['audio'];dur=r['duration_s'];core=au['official_core_s']>0;nonzero=not au['waveform_full_silence'];text=any(t.startswith('audio:') for t in r['counted_types'])
 for key,yes in [('official_core_positive',core),('waveform_nonzero',nonzero),('waveform_allzero',not nonzero),('text_audio_positive',text),('nonzero_without_text_audio',nonzero and not text),('nonzero_with_text_audio',nonzero and text),('text_audio_on_allzero',text and not nonzero)]:
  if yes:a[key+'_clips']+=1;a[key+'_clip_seconds']+=dur
 a['official_core_seconds']+=au['official_core_s'];a['nominal_keep_seconds']+=au['nominal_keep_s']
 flags.update(r.get('consistency_flags',[]))
 if i%50000==0:print(f'{i}/{n} {time.time()-start:.1f}s',flush=True)
if f:f.close()
assert i==n and a['waveform_nonzero_clips']+a['waveform_allzero_clips']==n
assert sum(c['clips'] for c in categories)==db.execute('SELECT SUM(types) FROM chosen').fetchone()[0]
result['audio']=dict(a);result['consistency_flags']=dict(flags)
result['definitions']=dict(category_names='Existing adopted open behavior names, not an adjudicated taxonomy or verified communicative backchannels',coverage_hours='Whole clip duration with a category; categories overlap, not action duration',nonzero='Stored processed-waveform identity flag; not verified listener vocalization',audio_seconds='Official union support / padded keep support, not measured vocal duration',read_scope='Existing statistics metadata only; no audio/video decoding; no labels changed')
(OUT/'overview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('categories',)},ensure_ascii=False,indent=2))
print('TOP',json.dumps(categories[:22],ensure_ascii=False));print('AUDIO',json.dumps([c for c in categories if c['type'].startswith('audio:')],ensure_ascii=False))
