"""Disjoint row execution of the unchanged OmniResponse backends and receipts."""
import sys,os,time,uuid,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,digest,job_key,artifact_valid,complete
from p2_omni_eval.pipeline import check_protocol,compute_row,has_error
from p2_omni_eval.backends import Backends
rank=int(sys.argv[1]);arm=['E0','E1','E0F','E1F'][rank//6];shard=rank%6;work=OUT/arm/'omni';p=read(work/'protocol.json');check_protocol(p)
backend=Backends(p,'cuda:0');results=[]
for row in p['rows'][shard::6]:
 key=job_key(row['sample_id'],row['sample_seed']);dest=work/'rows'/key;dest.mkdir(parents=True,exist_ok=True)
 ident=dict(protocol=digest(p),row=row,device='cuda:0')
 with (dest/'row.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if artifact_valid(dest,ident):record=read(dest/'result.json')
  else:
   attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
   record=compute_row(row,p,backend,attempt);atomic_json(attempt/'result.json',record);atomic_json(dest/'result.json',record)
   if not has_error(record):complete(dest,ident,[dest/'result.json',*attempt.glob('*.npy')])
  results.append(dict(sample_id=row['sample_id'],seed=row['sample_seed'],error=has_error(record)))
 atomic_json(work/f'shard-{shard}.json',dict(rank=rank,rows=results,finished=False,updated=time.time()))
check_protocol(p);atomic_json(work/f'shard-{shard}.json',dict(rank=rank,rows=results,finished=True,updated=time.time()))
