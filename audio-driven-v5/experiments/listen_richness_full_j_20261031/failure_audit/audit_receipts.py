from pathlib import Path
import json,sqlite3,time,concurrent.futures,collections
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');u=r/'experiments/listen_j_omni_20261030';w=r/'experiments/listen_richness_full_j_20261031'
def read(p):return json.loads(p.read_text())
q=read(u/'queues/caption/state.json');started=time.time();db=sqlite3.connect(u/'receipt_index.sqlite',timeout=60)
old=db.execute('SELECT offset FROM cursor WHERE id=1').fetchone();offset=old[0] if old else 0
with db,(u/'ready.jsonl').open() as f:
 f.seek(offset)
 while f.tell()<q['offset']:
  line=f.readline()
  if not line.endswith('\n'):break
  db.execute('INSERT OR IGNORE INTO sources(jid) VALUES(?)',(json.loads(line)['job_id'],));offset=f.tell()
 db.execute('INSERT OR REPLACE INTO cursor VALUES(1,?)',(offset,))
pending=[x[0] for x in db.execute('SELECT jid FROM sources WHERE done=0')]
def receipt(jid):
 p=u/'caption/completed'/f'{jid}.json'
 return jid,read(p) if p.exists() else None
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 for start in range(0,len(pending),256):
  with db:
   for jid,v in pool.map(receipt,pending[start:start+256]):
    if v is not None:db.execute('UPDATE sources SET done=1,caption=?,held=?,failed=?,visual_excluded=?,visual_review=? WHERE jid=?',(v['caption_clips'],v['held_clips'],v['failed_generation_clips'],v.get('visual_excluded_clips',0),v.get('visual_review_clips',0),jid))
fields=['done_sources','caption','held','generation_failed','visual_excluded','visual_review']
counts=dict(zip(fields,db.execute('SELECT SUM(done),SUM(caption),SUM(held),SUM(failed),SUM(visual_excluded),SUM(visual_review) FROM sources').fetchone()))
held=[]
for p in (u/'held'/'results').glob('*/*.json'):
 s=read(p);held.append(dict(sample_id=s['sample_id'],reason=s['reason'],file=str(p)))
reasons=collections.Counter(s['reason'] for s in held)
failed_sources=[x[0] for x in db.execute('SELECT jid FROM sources WHERE failed>0')];db.close()
report=dict(at=time.time(),seconds=time.time()-started,receipt_index_counts=counts,indexed_to_offset=offset,queue_snapshot_resolved=q['completed_clips'],held_file_counts=dict(reasons),held_files=len(held),generation_failed_sources=failed_sources,queue_failed=q['failed'])
audit=w/'failure_audit';audit.mkdir(exist_ok=True)
(audit/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(audit/'held_records.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in held))
policy=dict(version='exclude-nod-no-face-v1',at=time.time(),user_authorized=True,condition=dict(branch='held',reason_contains='no_detected_face'),action='exclude_from_training_and_caption_repair',preserve_original_files=True,other_held_reasons_unchanged=True)
(audit/'user_exclusion_policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2))
(audit/'excluded_nod_no_face.jsonl').write_text(''.join(json.dumps(dict(s,decision='exclude',policy=policy['version']),ensure_ascii=False)+'\n' for s in held if 'no_detected_face' in s['reason']))
print(json.dumps(report,ensure_ascii=False))
