#!/usr/bin/env python3
"""CPU-only AV500h re-selection from train+dev, per explicit scope correction.

Reuse the complete immutable v3 statistics. Retain each original source split.
The former train-only selection, labels, raw responses and caches stay frozen.
"""
import argparse
from collections import Counter,defaultdict
import csv,hashlib,json,math,sqlite3,sys,time
import orjson
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_j_upright_av500h_v3 as task
from run_j_upright_av500h_v3 import read,sha,ensure,jsonl,LABELS,EXPECTED_SHA,N,AUDIO_TYPES
from listen_richness.common import digest
from listen_richness.production import contract
from listen_benchmark.j_annotation import write
STATS=task.WORK
WORK=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105'
W=WORK

def initialize():
 WORK.mkdir(parents=True,exist_ok=True)
 prior=read(STATS/'config.json');delivery=read(STATS/'DELIVERY_COMPLETE.json')
 ensure(delivery['status']=='complete' and read(STATS/'STATISTICS_AUDIT.json')['passed'],'Unverified source statistics')
 ensure(prior['manifest_sha256']==EXPECTED_SHA and prior['contract']==contract(),'Wrong source statistics version')
 cfg=dict(schema='av500h-train-dev-source-scope-v1-20261105',project_root=str(ROOT),statistics_work=str(STATS),statistics_audit_sha256=sha(STATS/'STATISTICS_AUDIT.json'),manifest=str(LABELS/'labels.jsonl'),manifest_sha256=EXPECTED_SHA,contract=contract(),selection={k:v for k,v in prior['selection'].items() if k!='split'},source_split_policy='Use train+dev in both main and supplement; preserve original split; test excluded; no reannotation')
 cfg['selection']['source_splits']=['train','dev']
 p=WORK/'config.json'
 if p.exists():ensure(read(p)==cfg,'Existing corrected-scope config changed')
 else:write(p,cfg)
 write(WORK/'SCOPE_CORRECTION.json',dict(user_statement='不对，之前的筛选规则是从train+dev里面筛选',current_authoritative_pool=['train','dev'],previous_pool=['train'],historical_reference='related/P2_J数据筛选与500h有声补充方案记录_20261101.md has train+dev candidate overview and train-only old cache selection; current user instruction overrides the old cache scope',source_split_preserved=True,reused_all_1107313_statistics=True,new_model_inference=False,previous_selection_retained=str(STATS/'selection'),at=time.time()))

def describe(rows):
 c=Counter();hours=defaultdict(float);counts=Counter();durs=[];faces=[];modality=Counter()
 families={'点头':['visual:head nodding'],'微笑':['visual:smiling'],'眉部下压':['visual:eyebrow lowering','visual:brow lowering'],'说话/口头回应':['audio:speech'],'视线转移':['visual:gaze shifting'],'眨眼':['visual:blinking'],'摇头':['visual:head shaking'],'头部倾斜':['visual:head tilting'],'抬眉':['visual:eyebrow raising'],'其他发声':['audio:vocalization'],'可听笑声':['audio:audible laughter','audio:chuckling'],'哼声':['audio:humming']};fc=Counter()
 for r in rows:
  ts=set(r['counted_types']);c.update(ts);durs.append(r['duration_s']);faces.append(r['face_p10'])
  for t in ts:hours[t]+=r['duration_s']/3600
  for name,test in [('waveform_silent',r['audio_statistics']['waveform_full_silence']),('waveform_nonzero',not r['audio_statistics']['waveform_full_silence']),('text_audio_type',any(t.startswith('audio:') for t in ts)),('zero_adopted_types',not ts),('caption_review_flags',bool(r['review_flags'])),('orientation_restored','orientation_restoration' in r)]:
   if test:counts[name]+=1
  for fam,values in families.items():
   if ts.intersection(values):fc[fam]+=1
 for t in c:modality[t.split(':',1)[0]]+=1
 table=[dict(type=t,modality=t.split(':',1)[0],samples=n,sample_percent=n/len(rows)*100,covered_clip_hours=hours[t]) for t,n in c.most_common()]
 with (WORK/'selection/category_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['type','modality','samples','sample_percent','covered_clip_hours']);writer.writeheader();writer.writerows(table)
 counts['samples']=len(rows);durs.sort();faces.sort()
 result=dict(scope='Corrected train+dev AV500h selection; observable open labels, not validated functional backchannel taxonomy',counts=dict(counts),distinct_adopted_labels=len(c),distinct_labels_by_modality=dict(modality),total_sample_type_assignments=sum(c.values()),mean_types_per_sample=sum(c.values())/len(rows),source_split_counts=dict(Counter(r['split'] for r in rows)),top_types=table[:30],display_families=[dict(family=k,source_names=families[k],samples=n,sample_percent=n/len(rows)*100) for k,n in fc.most_common()],duration_s=dict(min=durs[0],median=durs[len(durs)//2],max=durs[-1],mean=sum(durs)/len(rows)),face_p10_px=dict(min=faces[0],median=faces[len(faces)//2]),occurrence_counts_are_reference_only=True)
 write(WORK/'selection/CATEGORY_SUMMARY.json',result)
 old=sqlite3.connect(f'file:{STATS}/selection/index.sqlite?mode=ro',uri=True);oldids={r[0] for r in old.execute('SELECT sid FROM samples')};old.close();newids={r['sample_id'] for r in rows}
 write(WORK/'selection/DIFF_FROM_TRAIN_ONLY.json',dict(previous_samples=len(oldids),new_samples=len(rows),common=len(oldids&newids),added=len(newids-oldids),removed=len(oldids-newids),added_by_source_split=dict(Counter(r['split'] for r in rows if r['sample_id'] not in oldids)),old_selection_unchanged=True))

def select():
 cfg=read(WORK/'config.json');ensure(read(STATS/'complete.json')['status']=='complete','Statistics not complete')
 out=WORK/'selection';out.mkdir(exist_ok=True);db=sqlite3.connect(f'file:{STATS}/tables/sample_stats.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 ensure(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==N,'Missing latest statistics')
 main=list(db.execute("SELECT * FROM samples WHERE split IN ('train','dev') AND face_p10>=224 ORDER BY sid"))
 main_seconds=sum(r['duration'] for r in main);needed=500*3600-main_seconds
 ensure(needed>0,'Main set already exceeds target; do not silently trim')
 marks=','.join('?' for _ in AUDIO_TYPES)
 candidates=list(db.execute(f"SELECT s.* FROM samples s WHERE s.split IN ('train','dev') AND s.face_p10>=184 AND s.face_p10<224 AND EXISTS(SELECT 1 FROM behaviors b WHERE b.sid=s.sid AND b.counted=1 AND b.type IN ({marks})) ORDER BY s.face_p10 DESC,s.face_p50 DESC,s.types DESC,s.sid ASC",AUDIO_TYPES))
 # Stream the combined statistics once; avoid hundreds of thousands of random
 # shared-filesystem opens while ranking supplements and binding the selection.
 wanted={r['sid'] for r in main}|{r['sid'] for r in candidates};stats={}
 with (STATS/'tables/sample_stats.jsonl').open('rb') as f:
  for b in f:
   r=orjson.loads(b);sid=r['sample_id']
   if sid not in wanted:continue
   ensure(sid not in stats,'Duplicate statistics stream ID')
   compact={k:r[k] for k in ('sample_id','caption_file_sha256','face_short_side_p10','record_status','counted_type_count','counted_types','behavior_quantities','audio')}
   compact.update(source_sha256=digest(r['source']),caption_text_sha256=hashlib.sha256(r['caption'].encode()).hexdigest());stats[sid]=compact
 ensure(set(stats)==wanted,'Statistics stream missing selection candidates')
 chosen=[];seconds=main_seconds;rejected_audio=0
 for row in candidates:
  r=stats[row['sid']]
  if r['audio']['waveform_full_silence'] or r['audio']['official_core_s']<=0:rejected_audio+=1;continue
  chosen.append(row);seconds+=row['duration']
  if seconds>=500*3600:break
 ensure(seconds>=500*3600,'Eligible supplements cannot reach 500h')
 lookup={r['sid']:('main_face224',r) for r in main}
 for r in chosen:ensure(r['sid'] not in lookup,'Duplicate selection');lookup[r['sid']]=('saved_audio_supplement',r)
 rows=[];seen=set();groups={};manifest_sha=hashlib.sha256();excluded={orjson.loads(b)['sample_id'] for b in (LABELS/'excluded.jsonl').read_bytes().splitlines()}
 for b in (LABELS/'labels.jsonl').open('rb'):
  manifest_sha.update(b);label=orjson.loads(b);g=label['group_id'];sp=label['split'];ensure(groups.setdefault(g,sp)==sp,'Group crosses split')
  sid=label['sample_id']
  if sid not in lookup:continue
  kind,stat=lookup[sid];ensure(label['split'] in ('train','dev') and sid not in excluded and sid not in seen,'Invalid selected membership');seen.add(sid);r=stats[sid]
  ensure(r['caption_file_sha256']==label['caption_file_sha256'] and r['source_sha256']==digest(label['source']) and r['caption_text_sha256']==hashlib.sha256(label['caption'].encode()).hexdigest(),'Stale statistics selected')
  ensure(math.isclose(r['face_short_side_p10'],label['face_p10'],abs_tol=1e-6,rel_tol=0),'Stale selected face measurement')
  if kind=='saved_audio_supplement':ensure(any(t in AUDIO_TYPES for t in r['counted_types']) and not r['audio']['waveform_full_silence'] and r['audio']['official_core_s']>0,'Supplement rule mismatch')
  rows.append(dict(label,source_split=label['split'],selection_group=kind,statistics_file=stat['file'],statistics_offset=stat['offset'],statistics_status=r['record_status'],counted_type_count=r['counted_type_count'],counted_types=r['counted_types'],behavior_quantities=r['behavior_quantities'],audio_statistics=r['audio'],selection_contract=cfg['selection']))
 ensure(manifest_sha.hexdigest()==EXPECTED_SHA and seen==set(lookup),'Manifest identity or coverage changed')
 rows.sort(key=lambda r:(r['selection_group']!='main_face224',r['sample_id']))
 info=jsonl(out/'selected.jsonl',rows)
 import csv
 fields=['sample_id','split','source_split','selection_group','duration_s','face_p10','face_p50','caption_file','caption_file_sha256','caption_identity_version','statistics_file','statistics_offset','counted_type_count']
 with (out/'sample_list.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 sql=out/'index.sqlite';ensure(not sql.exists(),'Do not overwrite a published selection index')
 ix=sqlite3.connect(sql);ix.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,selection_group TEXT,file TEXT,offset INTEGER,caption_sha256 TEXT,source_sha256 TEXT)')
 with (out/'selected.jsonl').open('rb') as f:
  while True:
   offset=f.tell();b=f.readline()
   if not b:break
   r=orjson.loads(b);ix.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(r['sample_id'],r['split'],r['selection_group'],str(out/'selected.jsonl'),offset,r['caption_file_sha256'],digest(r['source'])))
 ix.commit();ensure(ix.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==len(rows),'Selected index coverage mismatch');ix.close()
 summary=dict(status='complete',manifest_sha256=EXPECTED_SHA,statistics_contract=contract(),selection_contract=cfg['selection'],samples=len(rows),hours=seconds/3600,main=dict(samples=len(main),hours=main_seconds/3600),supplement=dict(samples=len(chosen),hours=(seconds-main_seconds)/3600,min_face_p10=min(r['face_p10'] for r in chosen)),rejected_audio_candidates_before_cutoff=rejected_audio,files={'selected.jsonl':info,'sample_list.csv':{'sha256':sha(out/'sample_list.csv')},'index.sqlite':{'sha256':sha(sql)}},source_splits=['train','dev'],source_split_counts=dict(Counter(r['split'] for r in rows)),hours_by_source_split={split:sum(r['duration_s'] for r in rows if r['split']==split)/3600 for split in ('train','dev')},test_excluded=True,no_split_group_leakage=True,all_selected_current_caption_and_source_bindings_verified=True,sft_filter_applied=False,caches_rebuilt=False,training_admitted=False,completed_at=time.time())
 describe(rows)
 write(out/'SUMMARY.json',summary);write(out/'COMPLETE.json',dict(status='complete',summary_sha256=sha(out/'SUMMARY.json'),selection_sha256=info['sha256'],samples=len(rows),hours=seconds/3600));print(json.dumps(summary),flush=True)

def audit():
 started=time.time();task.ensure(task.read(STATS/'STATISTICS_AUDIT.json')['passed'],'Statistics audit not complete');out=W/'selection';complete=task.read(out/'COMPLETE.json');summary=task.read(out/'SUMMARY.json');task.ensure(complete['summary_sha256']==task.sha(out/'SUMMARY.json'),'Selection summary SHA changed')
 db=sqlite3.connect(f'file:{STATS}/tables/sample_stats.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 main={r['sid']:dict(r) for r in db.execute("SELECT * FROM samples WHERE split IN ('train','dev') AND face_p10>=224")}
 candidates={r['sid']:dict(r) for r in db.execute("SELECT * FROM samples WHERE split IN ('train','dev') AND face_p10>=184 AND face_p10<224")};eligible=[]
 with (STATS/'tables/sample_stats.jsonl').open('rb') as f:
  for b in f:
   r=task.orjson.loads(b);sid=r['sample_id']
   if sid not in candidates:continue
   if set(task.AUDIO_TYPES).intersection(r['counted_types']) and r['audio']['waveform_full_silence'] is False and r['audio']['official_core_s']>0:eligible.append(candidates[sid])
 eligible.sort(key=lambda r:(-r['face_p10'],-r['face_p50'],-r['types'],r['sid']))
 duration=sum(r['duration'] for r in main.values());supp=[]
 for r in eligible:
  if duration>=500*3600:break
  supp.append(r);duration+=r['duration']
 expected=set(main)|{r['sid'] for r in supp};task.ensure(duration>=500*3600 and duration-500*3600<supp[-1]['duration']+1e-6,'Incorrect whole-clip 500h cutoff')
 selected={};offsets={};counts=Counter();fpath=out/'selected.jsonl'
 with fpath.open('rb') as f:
  h=hashlib.sha256()
  while True:
   offset=f.tell();b=f.readline()
   if not b:break
   h.update(b);r=task.orjson.loads(b);sid=r['sample_id'];task.ensure(sid not in selected,'Selected duplicate');selected[sid]=r;offsets[sid]=offset;counts[r['selection_group']]+=1;task.ensure(r['split'] in ('train','dev'),'Non-train selected')
  task.ensure(h.hexdigest()==complete['selection_sha256']==summary['files']['selected.jsonl']['sha256'],'Selected JSONL SHA differs')
 task.ensure(set(selected)==expected,'Independent ranking/main set differs')
 seen=set();manifest_main=set();manifest=hashlib.sha256();review=0;restored=0;hours=0
 with (task.LABELS/'labels.jsonl').open('rb') as f:
  for b in f:
   manifest.update(b);r=task.orjson.loads(b);sid=r['sample_id']
   if r['split'] in ('train','dev') and r['face_p10']>=224:manifest_main.add(sid)
   if sid not in selected:continue
   s=selected[sid];task.ensure(all(s[k]==v for k,v in r.items()),'Selected row changed original label fields: '+sid);seen.add(sid);hours+=r['duration_s']/3600;review+=bool(r['review_flags']);restored+='orientation_restoration' in r
   task.ensure(s['selection_group']==('main_face224' if sid in main else 'saved_audio_supplement'),'Incorrect selection group')
 task.ensure(manifest.hexdigest()==task.EXPECTED_SHA and seen==expected,'Selection/manifest coverage differs')
 task.ensure(manifest_main==set(main),'Main set is not the entire current manifest face>=224 train pool')
 ix=sqlite3.connect(f'file:{out}/index.sqlite?mode=ro',uri=True);index_ids=set()
 for sid,split,group,file,offset,caption_sha,source_sha in ix.execute('SELECT * FROM samples'):
  r=selected[sid];task.ensure(sid not in index_ids and split in ('train','dev') and split==r['split'] and group==r['selection_group'] and file==str(fpath) and offset==offsets[sid] and caption_sha==r['caption_file_sha256'] and source_sha==digest(r['source']),'Selection index binding differs');index_ids.add(sid)
 task.ensure(index_ids==expected,'Selected index coverage differs');ix.close()
 with (out/'sample_list.csv').open(encoding='utf-8-sig') as f:csvrows=list(csv.DictReader(f))
 task.ensure(len(csvrows)==len(expected) and {r['sample_id'] for r in csvrows}==expected,'CSV selected ID union differs')
 for r in csvrows:
  original=selected[r['sample_id']]
  task.ensure(all(value==('' if original[key] is None else str(original[key])) for key,value in r.items()),'Selected CSV field differs')
 task.ensure(math.isclose(hours,duration/3600,abs_tol=1e-7) and len(expected)==summary['samples'] and math.isclose(hours,summary['hours'],abs_tol=1e-7),'Selected duration/count differs')
 for filename in ('sample_list.csv','index.sqlite'):task.ensure(task.sha(out/filename)==summary['files'][filename]['sha256'],'Selected artifact changed')
 report=dict(passed=True,samples=len(expected),hours=hours,groups=dict(counts),ordinary_review_kept=review,orientation_restored_kept=restored,independent_python_ranking_matches=True,all_latest_label_fields_preserved=True,all_sqlite_offsets_and_source_shas_verified=True,csv_and_jsonl_index_coverage_matches=True,source_splits=['train','dev'],source_split_counts=dict(Counter(r['split'] for r in selected.values())),source_splits_preserved=True,test_excluded=True,full_statistics_audit_sha256=task.sha(STATS/'STATISTICS_AUDIT.json'),selection_complete_sha256=task.sha(out/'COMPLETE.json'),seconds=time.time()-started,at=time.time());write(out/'INDEPENDENT_FINAL_AUDIT.json',report);print(json.dumps(report))


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['select','audit','run']);args=p.parse_args();initialize()
 if args.command in ('select','run') and not (WORK/'selection/COMPLETE.json').exists():
  write(WORK/'RUN_STATUS.json',dict(stage='selecting_train_dev',at=time.time()));select()
 if args.command in ('audit','run'):
  write(WORK/'RUN_STATUS.json',dict(stage='independent_audit',at=time.time()));audit()
  summary=read(WORK/'selection/SUMMARY.json');check=read(WORK/'selection/INDEPENDENT_FINAL_AUDIT.json');ensure(check['passed'],'Independent audit failed')
  ensure(summary['source_split_counts']==check['source_split_counts'] and set(summary['source_split_counts'])=={'train','dev'},'Source split totals differ')
  write(WORK/'DELIVERY_COMPLETE.json',dict(status='complete',samples=summary['samples'],hours=summary['hours'],source_split_counts=summary['source_split_counts'],source_splits=['train','dev'],test_excluded=True,source_split_preserved=True,selection_summary_sha256=sha(WORK/'selection/SUMMARY.json'),independent_audit_sha256=sha(WORK/'selection/INDEPENDENT_FINAL_AUDIT.json'),source_statistics_audit_sha256=sha(STATS/'STATISTICS_AUDIT.json'),manifest_sha256=EXPECTED_SHA,model_inference_rerun=False,caches_rebuilt=False,completed_at=time.time()))
  write(WORK/'RUN_STATUS.json',dict(stage='complete',samples=summary['samples'],hours=summary['hours'],source_split_counts=summary['source_split_counts'],at=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  WORK.mkdir(parents=True,exist_ok=True);write(WORK/'ERROR.json',dict(error=repr(exc),at=time.time()));raise
