import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,digest,job_key,artifact_valid
from p2_omni_eval.pipeline import check_protocol,has_error
from p2_omni_eval.scoring import aggregate
from p2_omni_eval.compare import compare
for arm in ['E0','E1','E0F','E1F']:
 work=OUT/arm/'omni';p=read(work/'protocol.json');check_protocol(p);rows=[]
 assert all(read(work/f'shard-{i}.json')['finished'] for i in range(6))
 for row in p['rows']:
  dest=work/'rows'/job_key(row['sample_id'],row['sample_seed']);record=read(dest/'result.json')
  if not has_error(record):assert artifact_valid(dest,dict(protocol=digest(p),row=row,device='cuda:0'))
  rows.append(record)
 assert len({(r['sample_id'],r['sample_seed']) for r in rows})==48
 result=aggregate(p,rows);result.update(protocol_sha256=digest(p),device='cuda:0',execution='24 disjoint GPU workers; metrics from original backends, same full-set aggregation')
 atomic_json(work/'results.json',result);print(arm,result['status'],json.dumps(result['full_set_across_seeds']),flush=True)
for a,b in [('E0','E1'),('E0F','E1F'),('E0','E0F'),('E1','E1F')]:
 compare(OUT/a/'omni',OUT/b/'omni',OUT/f'omni_{a}_{b}.json')
