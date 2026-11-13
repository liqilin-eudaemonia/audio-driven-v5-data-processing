from pathlib import Path
import json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
W=Path(__file__).resolve().parent

def prepare():
 membership=read(W/'VALIDATION_MEMBERSHIP_ACCEPTED.json');chosen=read(W/'tuning64.json')
 assert sha(W/'tuning64.json')==membership['outputs']['tuning64.json']
 parent=ROOT/'experiments/seamless_quick200_latest_20261105/prepared_200';dst=W/'prepared64';dst.mkdir(exist_ok=True)
 ids={x['id']for x in chosen};refs=[json.loads(s)for s in(parent/'references.jsonl').read_text().splitlines()if s.strip()]
 refs=[r for r in refs if r['id']in ids];assert len(refs)==64
 text=''.join(json.dumps(x,ensure_ascii=False)+'\n'for x in refs);(dst/'references.jsonl').write_text(text)
 for name in ('source.test.jsonl','crop_config.json'):shutil.copy2(parent/name,dst/name)
 seal=read(parent/'seal.json');seal.update(expected_clips=64,parent_prepared=str(parent),parent_seal_sha256=sha(parent/'seal.json'),parent_references_sha256=sha(parent/'references.jsonl'),selected_ids_sha256=hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest(),references_sha256=sha(dst/'references.jsonl'),source_sha256=sha(dst/'source.test.jsonl'),crop_config_sha256=sha(dst/'crop_config.json'),selection_sha256=sha(W/'tuning64.json'))
 atomic_json(dst/'seal.json',seal,immutable=True)
 # Freeze requests now; no model export or GPU job until checkpoint exists.
 from p2_loss_search28.config import CANDIDATES,WORK
 for target in (500,1500,3000,6000):
  for name in CANDIDATES:
   out=W/'evaluations'/f'{name}_{target}';out.mkdir(parents=True,exist_ok=True)
   atomic_json(out/'REQUEST.json',dict(checkpoint=str(WORK/'runs'/name/f'step_{target:06d}.pt'),samples=64 if target==500 else 200),immutable=True)
 atomic_json(W/'SUBSET_PREPARED.json',dict(success=True,samples=64,seal_sha256=sha(dst/'seal.json'),references_sha256=sha(dst/'references.jsonl')))
if __name__=='__main__':prepare()
