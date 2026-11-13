from pathlib import Path
import sys,json,hashlib,time,csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_current_av_only.config import load,source_identity
from p2_current_av_only.data import Dataset
from p2_current_av_only.schedule import Schedule
out=Path(__file__).resolve().parent;c=load(out/'config.json');d=Dataset(c);s=Schedule(c);h=d.reader.source;t=d.reader.text
selection=Path(h.protocol['selection']);assert sha(selection)==d.index['selection_sha256']
def check(raw):
 row=json.loads(raw);sid=row['sample_id'];capraw=Path(row['caption_file']).read_bytes()
 assert hashlib.sha256(capraw).hexdigest()==row['caption_file_sha256'],sid
 cap=json.loads(capraw)
 for key in ('sample_id','caption','source','audio_recipe','crop_provenance','finish_reason','review_flags'):
  assert cap[key]==row[key],(sid,key)
 assert row['finish_reason']=='stop' and 'repeated_generation'not in row['review_flags'],sid
 for folder,expected in [(h.work/'cache/input_plan'/sid,h.identity(row,'input_plan')),(t.work/'cache/text_guided'/sid,t.identity(row))]:
  marker=read(folder/'complete.json')
  assert marker['contract']==expected and marker['identity']==digest(expected),str(folder)
  assert (folder/'payload.pt').stat().st_size>0,str(folder)
 return sid,row['split'],row['duration_s']
ids=[];splits=Counter();seconds=0.;began=time.time()
with selection.open('rb')as f,ThreadPoolExecutor(max_workers=8)as pool:
 from itertools import islice
 while lines:=list(islice(f,2048)):
  for sid,split,duration in pool.map(check,lines):ids.append(sid);splits[split]+=1;seconds+=duration
  atomic_json(out/'RECHECK_PROGRESS.json',dict(checked=len(ids),total=284218,elapsed=time.time()-began))
assert ids==[r['sample_id']for r in d.rows]==h.protocol['sample_ids']
assert dict(splits)=={'train':281060,'dev':3158} and abs(seconds/3600-d.index['hours'])<1e-6
for path in [out,R/'experiments/p2_current500h_av_only24_20261107']:
 cfg=load(path/'config.json');pre=read(path/'preflight.json');assert pre['source']==source_identity()
 Schedule(cfg)
atomic_json(out/'RECHECK_FULL.json',dict(success=True,at=time.time(),samples=len(ids),hours=seconds/3600,source_split_counts=dict(splits),
 selection=str(selection),selection_sha256=sha(selection),all_caption_files_hash_and_content_match=True,
 all_qwen_gemma_caption_contracts_match=True,cache_payload_checks='All active Qwen/Gemma metadata and nonempty payload presence; full tensor hashes remain checked at runtime, not reread here',
 source_identity_verified=True,profiles_verified=[24,32],elapsed_seconds=time.time()-began))
print(json.dumps(read(out/'RECHECK_FULL.json'),ensure_ascii=False,indent=2))
