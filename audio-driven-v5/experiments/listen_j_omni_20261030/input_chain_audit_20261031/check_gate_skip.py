import json,sys,time,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni import pending_caption_rows
from listen_visual_quality.io import canonical_hash
old=Path(read(W/'nod_cpu_share/config.json')['source_work']);hits={};scanned=0
# Inspect at most 24 currently assigned sources, not a result-directory walk.
q=read(old/'queues/nod/state.json')
for jid,owner in list(q['inflight'].items())[:24]:
 p=W/'visual_quality/early/completed'/f'{jid}.json'
 if not p.exists():continue
 receipt=read(p);job=read(old/'jobs'/f'{jid}.json');scanned+=1
 for decision in receipt['decisions']:
  kind=decision['decision']
  if kind not in ('exclude','review') or kind in hits:continue
  sid=decision['sample_id'];p=result_path(old,'nod',sid)
  if not p.exists():continue
  nod=read(p)
  if nod.get('status')!='visual_'+kind:continue
  assert nod['events'] is None and nod['visual_quality']==decision
  # A rejected current nod row has no detection pose or model inference data.
  assert 'pose' not in nod and 'scores' not in nod
  row=next(x for x in job['rows'] if x['sample_id']==sid)
  assert nod['source']==row['source'] and decision['source_sha256']==canonical_hash(row['source'])
  caption=result_path(W,'caption',sid)
  assert not caption.exists(),sid
  hits[kind]=dict(sample_id=sid,job_id=jid,nod_path=str(p),decision=decision)
 if len(hits)==2:break
# Exercise the exact admission function on copies; do not alter production.
import tempfile
with tempfile.TemporaryDirectory(prefix='jomni-gate-audit-') as tmp:
 w=Path(tmp)
 for kind,item in hits.items():
  job=read(old/'jobs'/f"{item['job_id']}.json");row=next(x for x in job['rows'] if x['sample_id']==item['sample_id'])
  todo,added=pending_caption_rows(w,dict(rows=[row]),'audit',{row['sample_id']:item['decision']})
  assert todo==[] and added[kind]==1
report=dict(at=time.time(),passed=len(hits)==2,sources_examined=scanned,examples=hits)
write(W/'input_chain_audit_20261031/gate_skip.json',report);print(json.dumps(dict(passed=report['passed'],sources=scanned,examples={k:v['sample_id'] for k,v in hits.items()})))
