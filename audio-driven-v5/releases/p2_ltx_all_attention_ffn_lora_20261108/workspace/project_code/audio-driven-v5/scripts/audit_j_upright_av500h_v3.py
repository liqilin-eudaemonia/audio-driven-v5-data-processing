#!/usr/bin/env python3
"""Independent read-only audit of extraction receipts and final AV500h membership."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_j_upright_av500h_v3 as task
from listen_richness.production import statistic_record,contract
from listen_richness.common import digest
from listen_benchmark.j_annotation import acquire,write
W=task.WORK

def watch():
 lock=acquire(W/'statistics-auditor.lock',False);task.ensure(lock is not None,'Statistics auditor already active')
 seen=set();groups={};jobs=set();workers=set();totals=Counter();started=time.time()
 with (W/'statistics_audit_chunks.jsonl').open('w') as ledger:
  while True:
   added=0
   for p in sorted((W/'extract/completed').glob('*.json')):
    if p.stem in jobs:continue
    r=task.read(p);j=task.read(W/'jobs'/f'{p.stem}.json')
    task.ensure(r['job_id']==j['job_id']==p.stem and r['contract']==j['contract']==contract(),'Receipt contract differs')
    for path,expected in [(j['path'],j['sha256']),(r['raw_file'],r['raw_sha256']),(r['statistics_file'],r['statistics_sha256'])]:task.ensure(task.sha(path)==expected,'Audited file SHA mismatch: '+path)
    reqs=[task.orjson.loads(b) for b in Path(j['path']).read_bytes().splitlines()];raws=[task.orjson.loads(b) for b in Path(r['raw_file']).read_bytes().splitlines()];stats=[task.orjson.loads(b) for b in Path(r['statistics_file']).read_bytes().splitlines()]
    task.ensure(len(reqs)==len(raws)==len(stats)==r['clips']==j['count'],'Receipt row count mismatch')
    task.ensure(r['input_ids_sha256']==digest([x['sample_id'] for x in reqs]),'Receipt input IDs changed')
    for i,(q,a,s) in enumerate(zip(reqs,raws,stats)):
     sid=q['sample_id'];snap=q['snapshot'];task.ensure(sid not in seen,'Duplicate audited sample: '+sid);seen.add(sid)
     task.ensure(sid==a['sample_id']==s['sample_id']==snap['sample_id'],'Response/metadata ID mismatch')
     task.ensure(digest(snap)==q['snapshot_sha256']==a['snapshot_sha256']==s['snapshot_sha256'],'Snapshot source SHA mismatch')
     task.ensure(s['caption_file_sha256']==snap['caption_file_sha256'] and s['caption']==snap['caption'] and s['source']==snap['source'],'Stale source/caption statistics')
     task.ensure(s['face_short_side_p10']==snap['face']['face_short_side_p10'] and s['audio']==snap['audio'],'Changed face/audio evidence')
     task.ensure(s['extraction_record_index']==i and s['extraction_file']==r['raw_file'],'Wrong raw response index')
     source=s['source'];task.ensure(groups.setdefault(source['group_id'],source['split'])==source['split'],'Cross-split group')
     task.ensure(s['retained'] and s['training_admitted'] is False,'Unexpected admission/removal')
     if s['record_status']=='unresolved_extraction':task.ensure(s['counted_type_count'] is None,'Unknown extraction became zero')
     for g in s['groups']:
      quantity=g['quantity']
      if quantity['basis'] in ('unknown','multiple_unspecified','partial_literal_not_total'):task.ensure(quantity['count'] is None,'Unknown count became known')
     totals['rows']+=1;totals['split_'+source['split']]+=1;totals['status_'+s['record_status']]+=1
     if snap.get('source_caption_review_flags'):totals['ordinary_review_preserved']+=1
     if snap.get('orientation_lineage'):totals['orientation_restored']+=1
    for i in sorted({0,len(reqs)-1}):
     replay=statistic_record(reqs[i],raws[i]['attempts'],r['raw_file']);actual=dict(stats[i]);actual.pop('extraction_record_index');task.ensure(replay==actual,'Frozen CPU aggregate replay differs')
    jobs.add(p.stem);workers.add((r['node'],r['gpu']));totals['replayed_boundary_rows']+=len({0,len(reqs)-1});added+=r['clips']
    ledger.write(json.dumps(dict(job_id=p.stem,clips=r['clips'],input_sha256=j['sha256'],raw_sha256=r['raw_sha256'],statistics_sha256=r['statistics_sha256'],node=r['node'],gpu=r['gpu']))+'\n');ledger.flush()
   write(W/'STATISTICS_AUDIT_PROGRESS.json',dict(audited_rows=totals['rows'],chunks=len(jobs),workers=len(workers),at=time.time()))
   if (W/'producer_complete.json').exists():
    pc=task.read(W/'producer_complete.json')
    if len(jobs)==pc['packed_chunks']:
     task.ensure(totals['rows']==task.N and totals['orientation_restored']==1690 and totals['ordinary_review_preserved']==130347,'Audited latest-label coverage differs')
     task.ensure({k:totals['split_'+k] for k in ('train','dev','test')}=={'train':1086207,'dev':8822,'test':12284},'Audited split totals differ')
     db=sqlite3.connect(f'file:{W}/inputs.sqlite?mode=ro',uri=True);expected={x[0] for x in db.execute("SELECT sid FROM coverage WHERE status='caption_ready'")};db.close();task.ensure(seen==expected,'Coverage/input ID union differs')
     ledger.flush();report=dict(passed=True,counts=dict(totals),chunks=len(jobs),workers=[list(x) for x in sorted(workers)],manifest_sha256=task.EXPECTED_SHA,frozen_contract=contract(),full_input_snapshot_and_statistics_hash_check=True,full_source_caption_face_audio_bindings=True,all_chunk_raw_response_sha_verified=True,first_last_each_chunk_cpu_replayed=True,unique_ids=len(seen),cross_split_groups=0,chunk_ledger_sha256=task.sha(W/'statistics_audit_chunks.jsonl'),seconds=time.time()-started,at=time.time());write(W/'STATISTICS_AUDIT.json',report);return
   if not added:time.sleep(10)

def final():
 started=time.time();task.ensure(task.read(W/'STATISTICS_AUDIT.json')['passed'],'Statistics audit not complete');out=W/'selection';complete=task.read(out/'COMPLETE.json');summary=task.read(out/'SUMMARY.json');task.ensure(complete['summary_sha256']==task.sha(out/'SUMMARY.json'),'Selection summary SHA changed')
 db=sqlite3.connect(f'file:{W}/tables/sample_stats.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 main={r['sid']:dict(r) for r in db.execute("SELECT * FROM samples WHERE split='train' AND face_p10>=224")}
 candidates={r['sid']:dict(r) for r in db.execute("SELECT * FROM samples WHERE split='train' AND face_p10>=184 AND face_p10<224")};eligible=[]
 with (W/'tables/sample_stats.jsonl').open('rb') as f:
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
 selected={};offsets={};counts=Counter();fpath=out/'train.jsonl'
 with fpath.open('rb') as f:
  h=hashlib.sha256()
  while True:
   offset=f.tell();b=f.readline()
   if not b:break
   h.update(b);r=task.orjson.loads(b);sid=r['sample_id'];task.ensure(sid not in selected,'Selected duplicate');selected[sid]=r;offsets[sid]=offset;counts[r['selection_group']]+=1;task.ensure(r['split']=='train','Non-train selected')
  task.ensure(h.hexdigest()==complete['selection_sha256']==summary['files']['train.jsonl']['sha256'],'Selected JSONL SHA differs')
 task.ensure(set(selected)==expected,'Independent ranking/main set differs')
 seen=set();manifest_main=set();manifest=hashlib.sha256();review=0;restored=0;hours=0
 with (task.LABELS/'labels.jsonl').open('rb') as f:
  for b in f:
   manifest.update(b);r=task.orjson.loads(b);sid=r['sample_id']
   if r['split']=='train' and r['face_p10']>=224:manifest_main.add(sid)
   if sid not in selected:continue
   s=selected[sid];task.ensure(all(s[k]==v for k,v in r.items()),'Selected row changed original label fields: '+sid);seen.add(sid);hours+=r['duration_s']/3600;review+=bool(r['review_flags']);restored+='orientation_restoration' in r
   task.ensure(s['selection_group']==('main_face224' if sid in main else 'saved_audio_supplement'),'Incorrect selection group')
 task.ensure(manifest.hexdigest()==task.EXPECTED_SHA and seen==expected,'Selection/manifest coverage differs')
 task.ensure(manifest_main==set(main),'Main set is not the entire current manifest face>=224 train pool')
 ix=sqlite3.connect(f'file:{out}/index.sqlite?mode=ro',uri=True);index_ids=set()
 for sid,split,group,file,offset,caption_sha,source_sha in ix.execute('SELECT * FROM samples'):
  r=selected[sid];task.ensure(sid not in index_ids and split=='train' and group==r['selection_group'] and file==str(fpath) and offset==offsets[sid] and caption_sha==r['caption_file_sha256'] and source_sha==digest(r['source']),'Selection index binding differs');index_ids.add(sid)
 task.ensure(index_ids==expected,'Selected index coverage differs');ix.close()
 with (out/'sample_list.csv').open(encoding='utf-8-sig') as f:csvrows=list(csv.DictReader(f))
 task.ensure(len(csvrows)==len(expected) and {r['sample_id'] for r in csvrows}==expected,'CSV selected ID union differs')
 for r in csvrows:
  original=selected[r['sample_id']]
  task.ensure(all(value==('' if original[key] is None else str(original[key])) for key,value in r.items()),'Selected CSV field differs')
 task.ensure(math.isclose(hours,duration/3600,abs_tol=1e-7) and len(expected)==summary['samples'] and math.isclose(hours,summary['hours'],abs_tol=1e-7),'Selected duration/count differs')
 for filename in ('sample_list.csv','index.sqlite'):task.ensure(task.sha(out/filename)==summary['files'][filename]['sha256'],'Selected artifact changed')
 report=dict(passed=True,samples=len(expected),hours=hours,groups=dict(counts),ordinary_review_kept=review,orientation_restored_kept=restored,independent_python_ranking_matches=True,all_latest_label_fields_preserved=True,all_sqlite_offsets_and_source_shas_verified=True,csv_and_jsonl_index_coverage_matches=True,train_only=True,full_statistics_audit_sha256=task.sha(W/'STATISTICS_AUDIT.json'),selection_complete_sha256=task.sha(out/'COMPLETE.json'),seconds=time.time()-started,at=time.time());write(out/'INDEPENDENT_FINAL_AUDIT.json',report);print(json.dumps(report))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['watch','final']);a=p.parse_args()
 try:watch() if a.command=='watch' else final()
 except BaseException as exc:write(W/'INDEPENDENT_AUDIT_ERROR.json',dict(error=repr(exc),at=time.time()));raise
