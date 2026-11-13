from pathlib import Path
import json,sqlite3,sys,time
from itertools import islice
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
out=Path(__file__).resolve().parent;c=read(out/'config.json');index=read(c['current_cache_index']);w=Path(c['teacher_cache']);p=read(w/'protocol.json')
assert sha(w/'mapping.sqlite')==p['mapping_sha256']
conn=sqlite3.connect((w/'mapping.sqlite').as_uri()+'?mode=ro',uri=True)
conn.execute('PRAGMA cache_size=-262144')
mapping_iter=iter(conn.execute('SELECT sid,record FROM samples ORDER BY rowid'))
selection=Path(read(Path(c['j_cache'])/'protocol.json')['selection']);assert sha(selection)==index['selection_sha256']
ph=sha(w/'protocol.json');increments=set(p['sample_ids']);counts=Counter();began=time.time();seen=set()
def check(pair):
 row,plan=pair;sid=row['sample_id'];kind=plan['teacher']
 assert plan['sample_id']==sid and plan['source_sha256']==digest(row['source']) and plan['split']==row['split']
 if kind=='reuse_both':
  assert sid not in increments
  f=Path(plan['teacher_old_payload']);m=read(f.parent/'complete.json')
  assert m['identity']==plan['teacher_identity'] and m['payload_sha256']==plan['teacher_payload_sha256']
  assert f.stat().st_size==plan['teacher_bytes']
 else:
  assert kind in ('new','reuse_audio') and sid in increments
  f=w/'cache'/sid/'payload.pt';m=read(f.parent/'complete.json')
  contract=dict(schema=p['schema'],protocol_sha256=ph,sample_id=sid,split=row['split'],source_sha256=digest(row['source']),audio_recipe_sha256=digest(row['audio_recipe']))
  assert m['contract']==contract and m['identity']==digest(contract) and f.stat().st_size>0
 return sid,kind
with selection.open()as stream,ThreadPoolExecutor(max_workers=16)as pool:
 while raw:=list(islice(stream,2048)):
  pairs=[]
  for line in raw:
   row=json.loads(line);mapped_sid,record=next(mapping_iter);assert mapped_sid==row['sample_id']
   pairs.append((row,json.loads(record)))
  for sid,kind in pool.map(check,pairs):assert sid not in seen;seen.add(sid);counts[kind]+=1
  atomic_json(out/'TEACHER_RECHECK_PROGRESS.json',dict(samples=len(seen),elapsed=time.time()-began))
assert next(mapping_iter,None) is None
assert len(seen)==284218 and counts==dict(reuse_both=276780,new=6542,reuse_audio=896)
atomic_json(out/'TEACHER_RECHECK_FULL.json',dict(success=True,samples=len(seen),counts=dict(counts),selection_sha256=sha(selection),elapsed=time.time()-began,verification='All selected media identities, mapping records, teacher cache contracts and payload presence/old sizes; full tensor SHA is checked by runtime load_part, not reread here'))
print(json.dumps(read(out/'TEACHER_RECHECK_FULL.json')))
