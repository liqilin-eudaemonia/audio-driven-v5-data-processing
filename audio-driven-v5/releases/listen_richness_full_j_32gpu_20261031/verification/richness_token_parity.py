import sys,json,time,concurrent.futures
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(r/'src'))
from transformers import AutoTokenizer
from listen_richness.production import tokenized,contract
from listen_richness.prepare_cluster import prepared_source,source_records
w=r/'experiments/listen_richness_full_j_20261031';rows=[json.loads(x) for x in (w/'benchmark_inputs.jsonl').read_text().splitlines()]
t=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True);out=[]
for n in (1,8):
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(n) as pool:result=list(pool.map(lambda row:tokenized(row['snapshot'],t),rows))
 assert result==rows
 out.append(dict(threads=n,rows=len(rows),seconds=time.monotonic()-start,identical=True))
print(out);(w/'token_parallel_validation.json').write_text(json.dumps(out,indent=2))
