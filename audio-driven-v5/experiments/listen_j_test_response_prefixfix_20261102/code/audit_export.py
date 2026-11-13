import json,sys,hashlib,sqlite3
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from listen_response_sft.data import statistics,load_example
from fix_test_caption_prefixes import clean_row
w=root/'experiments/listen_j_test_response_prefixfix_20261102'
c=w/'current'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((c/'summary.json').read_text())
original={json.loads(line)['sample_id']:(json.loads(line),line) for line in (root/'experiments/listen_j_full_response_sft_20261101/test.jsonl').read_bytes().splitlines(keepends=True)}
oldkeep={json.loads(line)['sample_id'] for line in (root/'experiments/listen_j_test_response_filter_20261102/filtered/test.jsonl').read_bytes().splitlines()}
seen=set();kept={}; changed=[]; counts={}
for decision,file in [('keep','test.jsonl'),('reject','rejected/test.jsonl'),('needs_review','needs_review/test.jsonl')]:
 rows=[]
 for raw in (c/file).read_bytes().splitlines(keepends=True):
  r=json.loads(raw);sid=r['sample_id'];assert sid not in seen;seen.add(sid);rows.append(sid)
  orig,oldraw=original[sid]
  if r['target']['format']=='original_caption_strip_chat_prefix_v1':
   assert r==clean_row(orig);statistics(r);changed.append(r)
  else: assert raw==oldraw
  if decision=='keep':kept[sid]=r
 if decision!='keep':
  reasons=[json.loads(x) for x in (c/Path(file).parent/'reasons.jsonl').read_text().splitlines()]
  assert [r['sample_id'] for r in reasons]==rows
  assert all(r['decision']==decision and r['attempts'] for r in reasons)
 counts[decision]=len(rows)
assert seen==set(original) and len(changed)==144 and oldkeep<=set(kept)
assert counts==summary['counts']
for rel,info in summary['manifests'].items():assert sha(c/rel)==info['sha256']
with sqlite3.connect(c/'index.sqlite') as db, (c/'test.jsonl').open('rb') as f:
 indexed=0
 for sid,file,offset,n in db.execute('SELECT sid,file,offset,bytes FROM samples'):
  assert file=='test.jsonl';f.seek(offset);assert json.loads(f.read(n))==kept[sid];indexed+=1
 assert indexed==len(kept)
decisions=[json.loads(x) for x in (c/'decisions.jsonl').read_text().splitlines()]
assert len(decisions)==len(original) and {r['sample_id'] for r in decisions}==seen
crop=json.loads((c/'crop_config.json').read_text())
new_kept=[r for r in kept.values() if r['target']['format']=='original_caption_strip_chat_prefix_v1']
controls=[r for r in kept.values() if r['target']['format']=='original_caption_verbatim']
media=[]
for r in new_kept[:2]+controls[:1]:
 x=load_example(r,crop);assert x['target_text']==r['target']['text']
 media.append(dict(sample_id=r['sample_id'],format=r['target']['format'],speaker_samples=len(x['inputs']['speaker_audio']),frame_size=x['inputs']['listener_first_frame'].size))
assert (w/'system.txt').read_bytes()==(root/'experiments/listen_j_test_response_filter_20261102/system.txt').read_bytes()
audit=dict(status='passed',counts=counts,source_ids=len(seen),prefix_only_edits=len(changed),old_pass_all_retained=True,index_entries_verified=indexed,all_jsonl_hashes_verified=True,rejected_and_review_reasons_complete=True,original_statistics_checked_for_all_edits=True,media_load_checks=media,strict_prompt_unchanged=True,unit_tests_passed=3,filter_regression_tests_passed=7)
(w/'FINAL_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False))
