import sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,prepare_shard,fuse_shard,ready_raw
run=Path(__file__).resolve().parent;raw=run/'raw_run';fusion=run/'fusion'
shards=list(read(raw/'queue/index.json')['shards'])
while not (run/'STOP').exists():
 ready=[s for s in shards if ready_raw(raw,s) and not (fusion/f'state/ingest/{s}.json').exists()]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(lambda s:prepare_shard(fusion,s),ready):print(result['shard'],flush=True)
 ingested=[s for s in shards if (fusion/f'state/ingest/{s}.json').exists()]
 if len(ingested)==len(shards):atomic(fusion/'INGEST_COMPLETE.json',dict(samples=512,shards=len(shards)))
 ready=[s for s in shards if (fusion/f'state/semantic/{s}.json').exists() and not (fusion/f'state/fusion/{s}.json').exists()]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(lambda s:fuse_shard(fusion,s),ready):print(result['shard'],result['counts'],flush=True)
 fused=[s for s in shards if (fusion/f'state/fusion/{s}.json').exists()]
 atomic(run/'pipeline_status.json',dict(ingested=len(ingested),fused=len(fused),total_shards=len(shards),updated_at=time.time()))
 if len(fused)==len(shards):break
 time.sleep(5)
