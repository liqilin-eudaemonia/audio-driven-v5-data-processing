import ast,collections,hashlib,json,sqlite3,sys,time
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from listen_response_sft.data import statistics,load_example
from fix_test_caption_prefixes import clean_row
from fix_train_dev_caption_prefixes import final_decision
w=root/'experiments/listen_j_train_dev_response_prefixfix_20261102'
c=w/'current';old=root/'experiments/listen_j_voiced_recovery_20261102/combined'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
s=read(c/'summary.json');cfg=read(w/'config.json')
assert read(w/'COMPLETE.json')['summary_sha256']==sha(c/'summary.json')
# Verify full retained manifests against the old combined manifests and exact new rows.
seen=set();rows_checked=0;cleaned_old=collections.Counter();new=collections.Counter();media_rows=[]
with sqlite3.connect(c/'index.sqlite') as db:
 for split in ['dev','train']:
  index=iter(db.execute('SELECT sid,file,offset,bytes FROM samples WHERE split=? ORDER BY rowid',(split,)))
  with (c/f'{split}.jsonl').open('rb') as f:
   for raw in (old/f'{split}.jsonl').open('rb'):
    original=json.loads(raw);offset=f.tell();current=f.readline();row=json.loads(current)
    normalized=clean_row(original)
    if normalized is not None:
     assert row==normalized;cleaned_old[split]+=1
    else:assert current==raw
    sid=row['sample_id'];assert sid not in seen;seen.add(sid)
    assert next(index)==(sid,f'{split}.jsonl',offset,len(current));rows_checked+=1
   for raw in (c/'recovered'/f'{split}.jsonl').open('rb'):
    offset=f.tell();current=f.readline();assert current==raw
    row=json.loads(raw);sid=row['sample_id'];assert sid not in seen;seen.add(sid)
    assert row['target']['format']=='original_caption_strip_chat_prefix_v1' and row['split']==split
    assert next(index)==(sid,f'{split}.jsonl',offset,len(current));rows_checked+=1;new[split]+=1
    if new[split]<=2:media_rows.append(row)
   assert not f.read(1) and next(index,None) is None
  assert cleaned_old[split]==cfg['counts'][split]['already_kept']
  assert new[split]==s['splits'][split]['counts']['new_keep']
 assert db.execute('SELECT count(*) FROM samples').fetchone()[0]==rows_checked==len(seen)
print('full manifest and index verification passed',rows_checked,flush=True)
# Every edit must still match its original statistics record; no media/crop rewriting.
edited_ids=set()
for split in ['dev','train']:
 for line in (w/'edited'/f'{split}.jsonl').open():
  row=json.loads(line);statistics(row);assert row['sample_id'] not in edited_ids;edited_ids.add(row['sample_id'])
assert len(edited_ids)==15033
records={}
for line in (c/'prefix_decisions.jsonl').open():
 r=json.loads(line);sid=r['sample_id'];assert sid not in records;records[sid]=r
 if r['route']!='previous_keep':
  assert final_decision(r['strict'],r['voiced'])==(r['decision'],r['route'])
  for j in [r['strict'],r['voiced']]:
   if j:assert j['attempts'] and j['caption_sha256']==hashlib.sha256(r['current_caption'].encode()).hexdigest()
assert set(records)==edited_ids
for split in ['dev','train']:
 for decision,folder in [('reject','rejected'),('needs_review','needs_review')]:
  ids=[json.loads(x)['sample_id'] for x in (c/folder/f'{split}.jsonl').open()]
  reasons=[json.loads(x) for x in (c/folder/f'{split}.reasons.jsonl').open()]
  assert ids==[r['sample_id'] for r in reasons]
  assert all(r==records[r['sample_id']] and r['decision']==decision for r in reasons)
  assert set(ids)=={sid for sid,r in records.items() if r['split']==split and r['decision']==decision}
for rel,meta in s['manifests'].items():assert sha(c/rel)==meta['sha256']
# Frozen prompts and decision functions must match the already used policies.
def funcs(p,names):
 return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(Path(p).read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names}
for name,previous,file,names in [
 ('strict',root/'experiments/listen_j_full_response_filter_20261101','driver.py',['schema','parse_answer','rule_evidence','static_visual_evidence']),
 ('voiced',root/'experiments/listen_j_voiced_recovery_20261102','driver.py',['schema','parse_answer','rule_evidence'])]:
 assert (w/name/'system.txt').read_bytes()==(previous/'system.txt').read_bytes()
 current='filter_full_response_sft.py' if name=='strict' else 'recover_voiced_response_sft.py'
 assert funcs(w/name/'code'/current,names)==funcs(previous/'code'/file,names)
media=[];crop=read(c/'crop_config.json')
for row in media_rows:
 x=load_example(row,crop);assert x['target_text']==row['target']['text']
 media.append(dict(sample_id=row['sample_id'],split=row['split'],speaker_samples=len(x['inputs']['speaker_audio']),image_size=x['inputs']['listener_first_frame'].size))
audit=dict(status='passed',all_previous_keeps_retained=True,previous_rows_unchanged_except_prefix=True,
 total_keep=rows_checked,all_index_entries_verified=True,cleaned_previous_keep=dict(cleaned_old),new_keep=dict(new),
 all_15033_edits_original_statistics_verified=True,all_affected_decisions_preserved=True,
 rejected_and_review_reasons_complete=True,all_manifest_hashes_verified=True,prompts_and_decision_functions_unchanged=True,
 media_load_checks=media,tests=dict(prefix=3,strict=7,voiced=5),test_processed=False,at=time.time())
(w/'FINAL_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False))
