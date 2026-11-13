"""Audit only the changed reader branches before pooling existing score blocks."""
import concurrent.futures,hashlib,json,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from listen_visual_quality.measurement import SourceBytes
import av

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old_sha='f1babf898eaaa6c0bf551fa984b0bae5a0274a72e3bee81f016bc102de87bc39'
new_sha='b56b68ee3837858a5ba71ac518f89d2e0bf207b6fa962b818653d5d6cfaf9819'
reader=ROOT.parent/'OmniResponse/evaluation/seamless/reader.py'
new=reader.read_text();original=new
a=new.index('def upright_reader_contract():');b=new.index('def frame_count',a);new=new[:a]+new[b:]
a=new.index("    if row.get('schema')=='j-upright-response-sft-reference-v1':");b=new.index('    return row,json.loads(config_path.read_text())',a);new=new[:a]+new[b:]
a=new.index("    if row.get('schema')=='j-upright-response-sft-reference-v1':");b=new.index('    return row,backend().resolve_example',a);new=new[:a]+new[b:]
a=new.index('    # Derived upright clips');b=new.index('    if len(frames)!=count:',a);new=new[:a]+new[b:]
a=new.index("    if row.get('schema')=='j-upright-response-sft-reference-v1':");b=new.index("    return example['inputs'],example['audit']",a);new=new[:a]+"    example=backend().load_example(row,cfg)\n"+new[b:]
assert hashlib.sha256(new.encode()).hexdigest()==old_sha
assert hashlib.sha256(original.encode()).hexdigest()==new_sha
assert sha(WORK/'reader_before_upright.py')==old_sha
base=read(WORK/'metrics/000/contract.json');groups={};contract_hashes={};audited_pairs=0;pair_sources=[]
for i in range(209):
 folder=WORK/f'metrics/{i:03d}';assert (folder/'SUCCESS.json').exists()
 contract=read(folder/'contract.json');contract_hashes[str(i)]=sha(folder/'contract.json')
 assert contract['assets']==base['assets']
 assert {k:v for k,v in contract['code'].items() if k!='../../seamless/reader.py'}=={k:v for k,v in base['code'].items() if k!='../../seamless/reader.py'}
 version=contract['code']['../../seamless/reader.py'];assert version in (old_sha,new_sha)
 for line in (folder/'pairs.jsonl').read_text().splitlines():
  r=json.loads(line);ref=r['input']['gt_reference']
  with open(ref['manifest'],'rb') as f:f.seek(ref['offset']);raw=f.read(ref['bytes'])
  assert hashlib.sha256(raw).hexdigest()==ref['row_sha256']
  row=json.loads(raw);assert row['schema']=='j-smallface-response-sft-reference-v1'
  if version==old_sha or 'gt_decoder' not in r.get('media',{}):continue
  d=r['media']['gt_decoder'];source=row['inputs']['listener_first_frame']['source'];start=row['inputs']['listener_first_frame']['source_time_s']
  assert abs(start-d['source_start_s'])<1e-8
  target=start+(d['frames']-1)/25
  key=json.dumps(source,sort_keys=True)
  if key not in groups or target>groups[key]['target']:groups[key]=dict(source=source,target=target,id=r['id'])
  pair_sources.append(dict(key=key,target=target,record=r))
  audited_pairs+=1
print('verify EOF branch unreachable:',audited_pairs,'pairs,',len(groups),'unique sources',flush=True)
def check(g):
 with SourceBytes(g['source']) as member,av.open(member) as container:
  stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
  container.seek(int(max(0,g['target']-1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
  last=None
  for frame in container.decode(video=0):
   if frame.pts is not None:last=float(frame.pts*frame.time_base)
   if frame.pts is not None and float(frame.pts*frame.time_base)+1e-8>=g['target']:
    return dict(id=g['id'],target=g['target'],available_pts=float(frame.pts*frame.time_base),source=g['source'])
 return dict(id=g['id'],target=g['target'],available_pts=last,source=g['source'],check_actual_old_reader=True)
checks=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 for future in concurrent.futures.as_completed([pool.submit(check,g) for g in groups.values()]):
  record=future.result()
  checks.append(record)
  if len(checks)%100==0:print('checked',len(checks),'/',len(groups),flush=True)
by_source={json.dumps(c['source'],sort_keys=True):c for c in checks}
import importlib.util
spec=importlib.util.spec_from_file_location('evaluation.seamless._frozen_reader',WORK/'reader_before_upright.py');old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
unavailable={};verified_old=[]
for pair in pair_sources:
 c=by_source[pair['key']]
 if not c.get('check_actual_old_reader') or (c['available_pts'] is not None and pair['target']<=c['available_pts']+1e-8):continue
 r=pair['record'];started=time.time()
 try:
  frames,fps,audit=old_module.decode_reference(r['input']['gt_reference'])
  assert audit['rgb_sha256']==r['media']['gt_sha256']
  verified_old.append(r['id'])
 except ValueError as exc:
  if not str(exc).startswith('Incomplete source decode:'):raise
  unavailable[r['id']]=dict(id=r['id'],identity=r['identity'],input=r['input'],metrics={},errors=dict(pair='ValueError: '+str(exc)),timings={},gt_cache_hits={},seconds=time.time()-started)
  print('native old-reader N/A',r['id'],str(exc),flush=True)
receipt=dict(passed=True,at=time.time(),reader_versions=[old_sha,new_sha],contracts_sha256=contract_hashes,
 proof='Exact old source reconstructed by removing only new upright-schema branches and EOF fallback; all requested rows use original schema; source PTS coverage proves the EOF branch unreachable for covered pairs; remaining pairs are checked with the exact frozen original reader, matching RGB hashes or retaining its native decode-error N/A. Existing raw scores remain unchanged; unavailable pairs are omitted from pooled features with an explicit audit record. Metric code and assets identical.',audited_pairs=audited_pairs,source_checks=checks,verified_old_reader_rgb_ids=verified_old,unavailable_records=unavailable)
t=WORK/'READER_COMPATIBILITY.tmp';t.write_text(json.dumps(receipt,indent=2)+'\n');t.replace(WORK/'READER_COMPATIBILITY.json')
print('COMPATIBLE',flush=True)
