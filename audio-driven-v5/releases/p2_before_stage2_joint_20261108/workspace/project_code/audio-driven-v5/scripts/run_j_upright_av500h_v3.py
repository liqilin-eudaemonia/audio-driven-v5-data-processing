#!/usr/bin/env python3
"""Audited manifest -> frozen v3 extraction -> historical AV500h selection.

Workers and aggregation retain the historical v3 implementation. This adapter
verifies the upright multi-hop lineage without modifying any original record.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import orjson
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import JobQueue,acquire,write
from listen_richness.common import digest
from listen_richness.production import contract,messages
from listen_richness.statistics import snapshot,audio_stats,face_stats,nod_stats
from listen_richness.prepare_cluster import database,publish
from listen_richness.merge_cluster import merge

WORK=ROOT/'experiments/listen_j_upright_av500h_v3_20261105'
LABELS=ROOT/'experiments/listen_j_labels_upright_20261105'
EXPECTED_SHA='0851714c52e17dec6c7b5145b82dc53998ab09af0cc7ceda0b9c807192c2b389'
N=1107313
IDENTITY=('sample_id','pair','split','group_id','listener_id','speaker_id','source_start_s','source_end_s','duration_s','normalized_audio')
AUDIO_TYPES=('audio:speech','audio:vocalization','audio:audible laughter','audio:humming','audio:chuckling')

def read(p):return orjson.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def ensure(ok,msg):
 if not ok:raise ValueError(msg)
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');h=hashlib.sha256();n=0
 with t.open('wb') as f:
  for row in rows:
   b=orjson.dumps(row)+b'\n';f.write(b);h.update(b);n+=1
  f.flush();os.fsync(f.fileno())
 t.replace(p);return dict(path=str(p),sha256=h.hexdigest(),count=n)

@lru_cache(maxsize=2048)
def verified_json(path,expected):
 b=Path(path).read_bytes();ensure(hashlib.sha256(b).hexdigest()==expected,'Evidence SHA mismatch: '+path);return orjson.loads(b)

def restored_snapshot(row,record):
 """Validate original -> original crop -> upright derivative -> final crop."""
 sid=row['sample_id'];o=row['orientation_restoration'];s=record['source'];original=record['original_source'];effective=record['original_effective_source'];d=row['quality'];cp=record['crop_provenance']
 rr=verified_json(o['restoration_file'],o['restoration_sha256'])
 prov=verified_json(o['provenance_file'],o['provenance_sha256'])
 ensure(rr['restoration_usable'] is True and record['orientation_restoration_file']==o['restoration_file'],'Unusable/wrong restoration')
 ensure(prov['original_source']==effective==o['original_effective_source'],'Restoration original effective source mismatch')
 ensure(prov['rotation_ccw_degrees']==o['rotation_ccw_degrees'] and o['rotation_ccw_degrees'] in (-90,90),'Wrong rotation')
 ensure(prov['timestamps_preserved'] and prov['external_audio_recipe_unchanged'] and prov['spatial_resize'] is False,'Restoration changed timeline/audio/scale')
 for k in IDENTITY:ensure(s[k]==original[k]==effective[k]==rr['request']['source'][k],sid+': changed '+k)
 ensure(original['source']==effective['source'] and original['normalized_audio']==s['normalized_audio'],'Original media/audio lineage differs')
 ensure(rr['parent_caption_sha256']==record['parent_caption_sha256']==row['regeneration_provenance']['parent_caption_sha256'],'Parent SHA differs')
 ensure(rr['request']['source']['source']==s['source'],'Derived archive differs')
 ensure(prov['derived_archive_sha256']==o['derived_archive_sha256']==s['orientation_restoration']['archive_sha256'],'Derived archive identity differs')
 ensure(Path(s['source']['archive']).is_file(),'Derived archive unavailable')
 measured=rr['request']['source'];ensure(d['source_sha256']==digest(measured,ascii=True),'Restored quality source mismatch')
 m=verified_json(d['measurement_file'],d['measurement_sha256'])
 ensure(m['source']['sample_id']==sid and m['source']==measured and m['source_sha256']==d['source_sha256'] and m['contract_sha256']==d['contract_sha256'] and m['status']=='ok','Restored measurement mismatch')
 ensure(cp['effective_source_sha256']==digest(s,ascii=True) and cp['original_source_sha256']==digest(measured,ascii=True),'Restored crop source SHA mismatch')
 ensure(cp['actual_crop']==s['video_transform']['crop'] and cp['original_crop']==measured['video_transform']['crop'],'Restored crop mismatch')
 ensure(cp['measurement_sha256']==d['measurement_sha256'] and cp['quality_contract_sha256']==d['contract_sha256'],'Restored crop measurement mismatch')
 sr=s['normalized_audio']['sample_rate'];duration=s['duration_s'];a=record['audio_recipe']['source']
 ensure(duration>0 and s['source_start_s']>=0 and abs(s['source_end_s']-s['source_start_s']-duration)<=1/sr,'Restored duration mismatch')
 ensure(abs((a['last_sample']-a['first_sample'])/sr-duration)<=1/sr,'Restored audio duration mismatch')
 e=record['input_evidence']
 return dict(sample_id=sid,caption=record['caption'],caption_file=row['caption_file'],caption_file_sha256=row['caption_file_sha256'],source=s,original_source_sha256=digest(original),quality=d,crop_provenance=cp,crop_revision=cp['revision'],face=face_stats(m),audio=audio_stats(record['audio_recipe'],e,s),nod=nod_stats(e['nod_supplement'],sid,duration),original_training_admitted=record.get('training_admitted'),training_admitted=False,human_ground_truth=False,orientation_lineage=dict(original_source_sha256=digest(original),original_effective_source_sha256=digest(effective),measurement_source_sha256=d['source_sha256'],effective_source_sha256=digest(s,ascii=True),restoration_file_sha256=o['restoration_sha256'],provenance_sha256=o['provenance_sha256'],archive_sha256=o['derived_archive_sha256']))

def from_manifest(row):
 """All raw captions/measurements are SHA verified; ordinary cue reviews stay."""
 raw=Path(row['caption_file']).read_bytes();ensure(hashlib.sha256(raw).hexdigest()==row['caption_file_sha256'],'Caption file changed: '+row['sample_id']);r=orjson.loads(raw)
 for k in ('sample_id','caption','source','audio_recipe','crop_provenance','finish_reason','review_flags'):
  ensure(r[k]==row[k],'Manifest/caption mismatch '+k+': '+row['sample_id'])
 ensure(row['finish_reason']=='stop' and row['caption'].strip(),'Incomplete/empty caption')
 ensure('repeated_generation' not in row['review_flags'],'Unrepaired repetition')
 ensure(row['quality']['decision']=='retain' and row['source']['listener_id']!=row['source']['speaker_id'],'Quality/roles mismatch')
 for k in IDENTITY:
  if k in row:ensure(row[k]==r['source'][k],'Manifest/source '+k+' mismatch')
 ensure(r['input_evidence']['listener_audio_is_silent']==row['listener_audio_is_silent'],'Recorded audio state differs')
 ensure(r['input_evidence']['nod_supplement']['detector_sha256']==row['detector_measurement_sha256'],'Detector SHA differs')
 s=restored_snapshot(row,r) if 'orientation_restoration' in row else snapshot(r,row['caption_file'],row['caption_file_sha256'],row['quality'],sample_grid_tolerance=True)
 for k,f in [('face_p10','face_short_side_p10'),('face_p50','face_short_side_p50')]:ensure(math.isclose(row[k],s['face'][f],rel_tol=0,abs_tol=1e-6),'Face measurement differs')
 # New policy distinguishes cue-review flags from hard generation failures.
 # Raw record and old helper remain unchanged; no quality failures are bypassed.
 if 'regeneration_provenance' in row:ensure(r['hard_generation_passed'] is True,'Regeneration hard failure')
 elif not row['generation_checks_passed']:ensure(bool(row['review_flags']),'Unexplained generation failure')
 s.update(eligible_for_extraction=True,exclusion_reasons=['caption_review_required'] if row['review_flags'] else [],source_caption_review_flags=row['review_flags'],source_generation_checks_passed=row['generation_checks_passed'],manifest_schema='upright-v3-source-adapter-20261105',caption_identity_version=row['caption_identity_version'])
 return s

def initialize():
 WORK.mkdir(parents=True,exist_ok=True)
 ensure(not (WORK/'config.json').exists(),'Already initialized; use existing work')
 old=read(ROOT/'experiments/listen_richness_full_j_20261031/config.json');hardware=read(ROOT/'experiments/listen_j_upright_regen_20261105/new_av500h_hardware_probe.json');nodes={}
 for h in hardware:
  ensure(h['rc']==0,'Node probe failed');name=h['node'];lines=h['out'].splitlines();uuids={}
  for x in lines[1:9]:
   i,u,mem,util=[s.strip() for s in x.split(',')];ensure(float(mem)<1024,'GPU occupied');uuids[i]=u
  ensure(len(uuids)==8,'Need eight GPUs per node');nodes[name]=dict(old['nodes'][name],host=h['host'],hostname=lines[0],gpu_uuids=uuids)
 ensure(contract()==old['contract'],'Not frozen historical v3')
 for n in ['inputs','jobs','prepare/jobs','prepare/prepare/completed','prepared','logs','tables']:(WORK/n).mkdir(parents=True,exist_ok=True)
 (WORK/'ready.jsonl').touch();(WORK/'prepare/ready.jsonl').touch()
 path=LABELS/'labels.jsonl';offset=0;start=0;count=0;total=0;whole=hashlib.sha256();block=hashlib.sha256();sources=[]
 with path.open('rb') as f,(WORK/'prepare/ready.jsonl').open('w') as journal:
  for line in f:
   whole.update(line);block.update(line);count+=1;total+=1;offset+=len(line)
   if count==2048:
    jid=f'manifest-{len(sources):05d}';job=dict(job_id=jid,path=str(path),offset=start,bytes=offset-start,sha256=block.hexdigest(),count=count);write(WORK/'prepare/jobs'/f'{jid}.json',job);journal.write(json.dumps({'job_id':jid})+'\n');sources.append(jid);start=offset;count=0;block=hashlib.sha256()
  if count:
   jid=f'manifest-{len(sources):05d}';job=dict(job_id=jid,path=str(path),offset=start,bytes=offset-start,sha256=block.hexdigest(),count=count);write(WORK/'prepare/jobs'/f'{jid}.json',job);journal.write(json.dumps({'job_id':jid})+'\n');sources.append(jid)
 ensure(whole.hexdigest()==EXPECTED_SHA and total==N,'Latest immutable manifest changed')
 export=read(LABELS/'EXPORT_COMPLETE.json');ensure(export['status']=='complete','Manifest unpublished')
 paths=list(old['code_hashes'])+['scripts/run_j_upright_av500h_v3.py','scripts/control_j_upright_av500h_v3.py'];hashes={p:sha(ROOT/p) for p in paths}
 cfg=dict(schema='upright-av500h-frozen-v3-20261105',project_root=str(ROOT),upstream=str(LABELS),manifest=str(path),manifest_sha256=EXPECTED_SHA,nodes=nodes,contract=contract(),code_hashes=hashes,profiles=old['profiles'],selected_profile='D512',chunk_size=512,prepare_workers_per_node=32,expected_upstream_clips=1107323,expected_upstream_sources=len(sources),expected_labels=N,prepare_sources=sources,selection=dict(split='train',main_face_p10_min=224,supplement_face_p10_min=184,supplement_face_p10_max_exclusive=224,audio_types=list(AUDIO_TYPES),waveform_nonzero=True,official_core_positive=True,target_hours=500,sort=['face_p10 DESC','face_p50 DESC','types DESC','sid ASC']),created_at=time.time())
 write(WORK/'config.json',cfg);write(WORK/'INPUT_MANIFEST_VERIFIED.json',dict(sha256=EXPECTED_SHA,labels=total,bytes=offset,source_blocks=len(sources),frozen_contract=contract(),export_complete_sha256=sha(LABELS/'EXPORT_COMPLETE.json'),upright_audit_sha256=sha(LABELS/'UPRIGHT_AUDIT.json')))
 print(json.dumps(dict(stage='initialized',labels=total,blocks=len(sources),contract=contract())),flush=True)

def prepare(node):
 from transformers import AutoTokenizer
 cfg=read(WORK/'config.json');lock=acquire(WORK/'producer_locks'/f'{node}.lock',False);ensure(lock is not None,'Producer already active')
 tokenizer=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True)
 q=JobQueue(WORK/'prepare','prepare');done=0
 with ThreadPoolExecutor(cfg['prepare_workers_per_node']) as pool:
  while found:=q.claim(node):
   jid,handle=found
   try:
    job=read(WORK/'prepare/jobs'/f'{jid}.json')
    with Path(job['path']).open('rb') as f:f.seek(job['offset']);raw=f.read(job['bytes'])
    ensure(hashlib.sha256(raw).hexdigest()==job['sha256'],'Manifest block changed')
    rows=[orjson.loads(b) for b in raw.splitlines()];ensure(len(rows)==job['count'],'Block count differs')
    snaps=list(pool.map(from_manifest,rows))
    reqs=[]
    # Batch the unchanged chat templates; fast tokenizer parallelism on CPU.
    for first in range(0,len(snaps),64):
     batch=snaps[first:first+64];ids=tokenizer.apply_chat_template([messages(s['caption']) for s in batch],tokenize=True,add_generation_prompt=True,enable_thinking=False)
     ensure(len(ids)==len(batch),'Token batch count differs')
     for s,tokens in zip(batch,ids):
      ensure(len(tokens)+8192<=12288,'Context overflow; no truncation')
      from listen_richness.extraction import sentences
      reqs.append(dict(sample_id=s['sample_id'],snapshot_sha256=digest(s),snapshot=s,prompt_token_ids=tokens,prompt_tokens=len(tokens),sentence_count=len(sentences(s['caption']))))
    reqs.sort(key=lambda r:(-r['prompt_tokens'],r['sample_id']));chunks=[]
    for first in range(0,len(reqs),512):
     cj=f'{jid}-{first//512:02d}';info=jsonl(WORK/'inputs'/f'{cj}.jsonl',reqs[first:first+512]);info.update(job_id=cj,contract=contract());write(WORK/'jobs'/f'{cj}.json',info);chunks.append(info)
    coverage=[]
    for row in rows:coverage.append(dict(sample_id=row['sample_id'],job_id=jid,status='caption_ready',split=row['split'],group_id=row['group_id'],caption_file=row['caption_file'],caption_file_sha256=row['caption_file_sha256'],caption_identity_version=row['caption_identity_version'],source_sha256=digest(row['source']),review_flags=row['review_flags']))
    ci=jsonl(WORK/'prepared'/f'{jid}.coverage.jsonl',coverage)
    receipt=dict(job_id=jid,count=len(rows),node=node,chunks=chunks,coverage=ci,manifest_block=job,completed_at=time.time())
    write(WORK/'prepared'/f'{jid}.json',receipt);q.finish(jid,receipt=receipt);done+=len(rows)
    write(WORK/'producers'/f'{node}.json',dict(stage='preparing',done=done,last_job=jid,at=time.time()))
   except Exception as exc:q.finish(jid,error=repr(exc));write(WORK/'producers'/f'{node}.json',dict(stage='error',error=repr(exc),job_id=jid,done=done));raise
   finally:handle.close()
 write(WORK/'producers'/f'{node}.json',dict(stage='complete',done=done,at=time.time()));lock.close()

def coordinate():
 owner=acquire(WORK/'coordinator.lock',False);ensure(owner is not None,'Coordinator already active')
 cfg=read(WORK/'config.json');db=database(WORK)
 for jid in cfg['prepare_sources']:db.execute("INSERT OR IGNORE INTO sources(jid,status) VALUES (?,'pending')",(jid,))
 # Terminal non-caption records retain coverage, never receive fake statistics.
 for filename,status in [('excluded.jsonl','visual_excluded'),('exceptions.jsonl','held')]:
  for b in (LABELS/filename).read_bytes().splitlines():
   r=orjson.loads(b);base=dict(r,status=status)
   db.execute('INSERT OR IGNORE INTO coverage VALUES (?,?,?,?,?)',(r['sample_id'],'terminal',status,None,json.dumps(base)))
 db.commit()
 try:
  while True:
   pending=db.execute("SELECT jid FROM sources WHERE status!='prepared'").fetchall()
   if not pending:break
   added=0
   for (jid,) in pending:
    p=WORK/'prepared'/f'{jid}.json'
    if not p.exists():continue
    r=read(p);ensure(r['manifest_block']==read(WORK/'prepare/jobs'/f'{jid}.json'),'Prepared source contract differs')
    ensure(sha(r['coverage']['path'])==r['coverage']['sha256'],'Prepared coverage changed')
    rows=[orjson.loads(b) for b in Path(r['coverage']['path']).read_bytes().splitlines()]
    mapping={}
    for cj in r['chunks']:
     ensure(cj['contract']==contract(),'Prepared prompt contract differs')
     ensure(sha(cj['path'])==cj['sha256'],'Packed request changed')
     for b in Path(cj['path']).read_bytes().splitlines():
      x=orjson.loads(b);ensure(x['sample_id'] not in mapping,'Duplicate source IDs');mapping[x['sample_id']]=cj['job_id']
    ensure(len(mapping)==len(rows)==r['count'],'Prepared coverage cardinality mismatch')
    with db:
     for row in rows:db.execute('INSERT INTO coverage VALUES (?,?,?,?,?)',(row['sample_id'],jid,'caption_ready',mapping[row['sample_id']],json.dumps(row,ensure_ascii=False)))
     for cj in r['chunks']:db.execute('INSERT INTO chunks(jid,n) VALUES (?,?)',(cj['job_id'],cj['count']))
     db.execute("UPDATE sources SET status='prepared' WHERE jid=?",(jid,))
    publish(db,WORK);added+=r['count']
   cov=dict(db.execute('SELECT status,COUNT(*) FROM coverage GROUP BY status'));source_count=db.execute("SELECT COUNT(*) FROM sources WHERE status='prepared'").fetchone()[0]
   write(WORK/'preparation.json',dict(at=time.time(),coverage=cov,prepared_blocks=source_count,total_blocks=len(cfg['prepare_sources']),new_labels=added))
   if not added:time.sleep(3)
  cov=dict(db.execute('SELECT status,COUNT(*) FROM coverage GROUP BY status'))
  ensure(cov=={'caption_ready':N,'held':1,'visual_excluded':9},'Incomplete manifest coverage')
  ensure(db.execute('SELECT SUM(n) FROM chunks').fetchone()[0]==N,'Input chunk count mismatch')
  write(WORK/'producer_complete.json',dict(at=time.time(),coverage=cov,packed_chunks=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0],manifest_sha256=EXPECTED_SHA))
 finally:db.close();owner.close()

def select():
 cfg=read(WORK/'config.json');ensure(read(WORK/'complete.json')['status']=='complete','Statistics not complete')
 out=WORK/'selection';out.mkdir(exist_ok=True);db=sqlite3.connect(f'file:{WORK}/tables/sample_stats.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 ensure(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==N,'Missing latest statistics')
 main=list(db.execute("SELECT * FROM samples WHERE split='train' AND face_p10>=224 ORDER BY sid"))
 main_seconds=sum(r['duration'] for r in main);needed=500*3600-main_seconds
 ensure(needed>0,'Main set already exceeds target; do not silently trim')
 marks=','.join('?' for _ in AUDIO_TYPES)
 candidates=list(db.execute(f"SELECT s.* FROM samples s WHERE s.split='train' AND s.face_p10>=184 AND s.face_p10<224 AND EXISTS(SELECT 1 FROM behaviors b WHERE b.sid=s.sid AND b.counted=1 AND b.type IN ({marks})) ORDER BY s.face_p10 DESC,s.face_p50 DESC,s.types DESC,s.sid ASC",AUDIO_TYPES))
 # Stream the combined statistics once; avoid hundreds of thousands of random
 # shared-filesystem opens while ranking supplements and binding the selection.
 wanted={r['sid'] for r in main}|{r['sid'] for r in candidates};stats={}
 with (WORK/'tables/sample_stats.jsonl').open('rb') as f:
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
  kind,stat=lookup[sid];ensure(label['split']=='train' and sid not in excluded and sid not in seen,'Invalid selected membership');seen.add(sid);r=stats[sid]
  ensure(r['caption_file_sha256']==label['caption_file_sha256'] and r['source_sha256']==digest(label['source']) and r['caption_text_sha256']==hashlib.sha256(label['caption'].encode()).hexdigest(),'Stale statistics selected')
  ensure(math.isclose(r['face_short_side_p10'],label['face_p10'],abs_tol=1e-6,rel_tol=0),'Stale selected face measurement')
  if kind=='saved_audio_supplement':ensure(any(t in AUDIO_TYPES for t in r['counted_types']) and not r['audio']['waveform_full_silence'] and r['audio']['official_core_s']>0,'Supplement rule mismatch')
  rows.append(dict(label,selection_group=kind,statistics_file=stat['file'],statistics_offset=stat['offset'],statistics_status=r['record_status'],counted_type_count=r['counted_type_count'],counted_types=r['counted_types'],behavior_quantities=r['behavior_quantities'],audio_statistics=r['audio'],selection_contract=cfg['selection']))
 ensure(manifest_sha.hexdigest()==EXPECTED_SHA and seen==set(lookup),'Manifest identity or coverage changed')
 rows.sort(key=lambda r:(r['selection_group']!='main_face224',r['sample_id']))
 info=jsonl(out/'train.jsonl',rows)
 import csv
 fields=['sample_id','split','selection_group','duration_s','face_p10','face_p50','caption_file','caption_file_sha256','caption_identity_version','statistics_file','statistics_offset','counted_type_count']
 with (out/'sample_list.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 sql=out/'index.sqlite';ensure(not sql.exists(),'Do not overwrite a published selection index')
 ix=sqlite3.connect(sql);ix.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,selection_group TEXT,file TEXT,offset INTEGER,caption_sha256 TEXT,source_sha256 TEXT)')
 with (out/'train.jsonl').open('rb') as f:
  while True:
   offset=f.tell();b=f.readline()
   if not b:break
   r=orjson.loads(b);ix.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(r['sample_id'],r['split'],r['selection_group'],str(out/'train.jsonl'),offset,r['caption_file_sha256'],digest(r['source'])))
 ix.commit();ensure(ix.execute('SELECT COUNT(*) FROM samples').fetchone()[0]==len(rows),'Selected index coverage mismatch');ix.close()
 summary=dict(status='complete',manifest_sha256=EXPECTED_SHA,statistics_contract=contract(),selection_contract=cfg['selection'],samples=len(rows),hours=seconds/3600,main=dict(samples=len(main),hours=main_seconds/3600),supplement=dict(samples=len(chosen),hours=(seconds-main_seconds)/3600,min_face_p10=min(r['face_p10'] for r in chosen)),rejected_audio_candidates_before_cutoff=rejected_audio,files={'train.jsonl':info,'sample_list.csv':{'sha256':sha(out/'sample_list.csv')},'index.sqlite':{'sha256':sha(sql)}},all_train=True,no_split_group_leakage=True,all_selected_current_caption_and_source_bindings_verified=True,sft_filter_applied=False,caches_rebuilt=False,training_admitted=False,completed_at=time.time())
 write(out/'SUMMARY.json',summary);write(out/'COMPLETE.json',dict(status='complete',summary_sha256=sha(out/'SUMMARY.json'),selection_sha256=info['sha256'],samples=len(rows),hours=seconds/3600));print(json.dumps(summary),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','prepare','coordinate','merge','select']);p.add_argument('--node');a=p.parse_args()
 if a.command=='init':initialize()
 elif a.command=='prepare':prepare(a.node)
 elif a.command=='coordinate':coordinate()
 elif a.command=='merge':print(json.dumps(merge(WORK)))
 elif a.command=='select':select()
if __name__=='__main__':main()
