import sys,os,json,time,concurrent.futures
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(root/'src'))
from listen_richness.prepare_cluster import source_records
w=root/'experiments/listen_richness_full_j_20261031';u=root/'experiments/listen_j_omni_20261030'
rows=[json.loads(x) for x in (w/'benchmark_inputs.jsonl').read_text().splitlines()]
# Fixed 64 indexed sources already used in the benchmark, no directory traversal.
ids=[]
for r in rows:
 s=r['snapshot']['caption_file'];record=json.loads(Path(s).read_text());
 # source id comes from coverage index
import sqlite3
with sqlite3.connect(f'file:{w}/inputs.sqlite?mode=ro',uri=True) as db:
 ids=[r[0] for r in db.execute("SELECT jid FROM sources WHERE status='prepared' ORDER BY jid LIMIT 64")]
out=[]
for workers in (8,16,32,16):
 t=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(workers) as pool:
  results=list(pool.map(lambda jid:source_records(u,jid),ids))
 n=sum(len(r) for r in results);dt=time.monotonic()-t
 out.append(dict(workers=workers,clips=n,seconds=dt,clips_per_second=n/dt));print(out[-1],flush=True)
(w/'prepare_read_benchmark.json').write_text(json.dumps(out,indent=2))
