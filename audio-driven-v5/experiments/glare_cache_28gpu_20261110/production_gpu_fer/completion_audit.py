import pathlib,json,hashlib,time
R=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5');b=R/'experiments/glare_cache_28gpu_20261110/production_gpu_fer'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
latest=read(b/'TRAINING_CACHES_LATEST.json');out=pathlib.Path(latest['export'])
assert sha(out/'SEAL.json')==latest['seal_sha256']
seal=read(out/'SEAL.json');checked={}
for n,h in seal['files'].items():assert sha(out/n)==h;checked[n]=h
assert sha(b/'manifest.json')==seal['manifest_sha256']
summary=read(out/'SUMMARY.json');assert summary['cache_registry_complete'] and summary['unavailable']==0
members={};counts={};repaired=False
for split in ['train','dev']:
 counts[split]=0
 with (out/(split+'.jsonl')).open() as f:
  for raw in f:
   row=json.loads(raw);sid=row['sample_id'];assert sid not in members and row['split']==split
   assert set(row['glare_caches'])=={'lia','audio','statistics','reaction','frozen_audio','prosody','view'}
   members[sid]=split;counts[split]+=1
assert counts=={'train':281060,'dev':3158}
selection=R/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection/selected.jsonl'
original={}
with selection.open() as f:
 for raw in f:
  row=json.loads(raw);assert row['sample_id'] not in original;original[row['sample_id']]=row['split']
assert members==original
for stage in ['lia','prosody']:
 paths=list((b/'tasks'/stage).glob('*/done.json'));assert len(paths)==224
assert read(b/'tasks/lia/0122/done.json')['result']['validated']
final=read(b/'finalize_status.json');assert final['stage']=='complete' and final['returncode']==0
status=read(b/'status.json');assert all(n['stage']=='complete' for n in status['nodes'].values())
result=dict(at=time.time(),passed=True,counts=counts,pairs=len(members),selection_members_and_splits_exact=True,registry_seal_and_file_hashes_verified=checked,all_seven_caches_per_row=True,unavailable=0,task_receipts_per_stage=224,finalizer_exit=0,node_supervisors_complete=True,full_payload_hashes_reread=False,note='Assembler verified every payload exists/nonempty and receipts/identity; no second full reread of tensor bytes. Training reader checks sampled real tensors separately.')
(b/'COMPLETION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
