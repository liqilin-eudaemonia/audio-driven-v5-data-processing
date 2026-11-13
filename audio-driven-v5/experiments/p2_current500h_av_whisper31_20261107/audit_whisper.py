from pathlib import Path
import json,sys,time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from collections import Counter
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_j_current_qwen_hidden import item_from_row
from p2_whisper_av.cache import CACHE_VERSION,observation_contract
out=Path(__file__).resolve().parent;c=read(out/'config.json');w=Path(c['whisper_cache']);p=read(w/'protocol.json');hp=read(Path(c['j_cache'])/'protocol.json');mapping=R/'experiments/p2_j_train_dev500h_aux_reuse_audit_20261105/reuse_map.jsonl'
assert sha(mapping)==p['reuse_map_sha256'] and sha(hp['selection'])==p['selection_sha256']
new=set(p['sample_ids']);counts=Counter();seen=set();began=time.time()
def check(pair):
 row,plan=pair;sid=row['sample_id'];assert plan['sample_id']==sid and plan['source_sha256']==digest(row['source']) and plan['split']==row['split']
 if sid in new:
  assert plan['whisper']=='new';folder=w/'cache'/sid;m=read(folder/'complete.json')
  contract=dict(schema=CACHE_VERSION,code=p['code'],encoder=p['encoder'],observation=observation_contract(item_from_row(row)))
  assert m['contract']==contract and m['identity']==digest(contract)
  assert (folder/'payload.pt').stat().st_size>0
  return sid,'new'
 assert plan['whisper']=='reuse';f=Path(plan['whisper_old_payload']);m=read(f.parent/'complete.json');o=m['contract']['observation'];src=row['source']
 assert m['identity']==plan['whisper_identity']==digest(m['contract']) and m['payload_sha256']==plan['whisper_payload_sha256']
 assert f.stat().st_size==plan['whisper_bytes'] and digest(o)==plan['current_observation_sha256']
 assert all(o[k]==src[k] for k in ('speaker_id','listener_id','source_start_s','source_end_s'))
 assert o['sample_id']==sid and o['origin_s']==0 and m['contract']['encoder']==p['encoder']
 return sid,'reuse'
with Path(hp['selection']).open()as f,mapping.open()as g,ThreadPoolExecutor(max_workers=16)as pool:
 while raw:=list(islice(f,2048)):
  pairs=[(json.loads(x),json.loads(next(g)))for x in raw]
  for sid,kind in pool.map(check,pairs):assert sid not in seen;seen.add(sid);counts[kind]+=1
  atomic_json(out/'WHISPER_AUDIT_PROGRESS.json',dict(samples=len(seen),elapsed=time.time()-began))
 assert next(g,None)is None
assert counts=={'new':6542,'reuse':277676} and len(seen)==284218
atomic_json(out/'WHISPER_AUDIT_FULL.json',dict(success=True,samples=len(seen),counts=dict(counts),selection_sha256=p['selection_sha256'],elapsed_seconds=time.time()-began,verification='All speaker identities, intervals, reused observation contracts, increment contracts and payload presence; runtime rehashes tensor payloads'))
print(json.dumps(read(out/'WHISPER_AUDIT_FULL.json')))
