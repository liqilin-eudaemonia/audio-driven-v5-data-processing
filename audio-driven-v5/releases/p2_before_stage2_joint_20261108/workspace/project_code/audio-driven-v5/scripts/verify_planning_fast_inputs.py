"""Compare exact training inputs and processor tensors before reader migration."""
import argparse,json,sys,time,sqlite3,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from p2_listen.contracts import read,sha,atomic_json
from p2_planning_sft.reference_data import ReferencePlanningDataset
from p2_planning_sft.data import PlanningCollator


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 c=read(a.config);torch.set_num_threads(2);base=ReferencePlanningDataset(c['data']['seal'],'train')
 fast=ReferencePlanningDataset(c['data']['seal'],'train',input_pipeline=c['data']['input_pipeline'])
 rng=random.Random(20261102);ids=rng.sample(base.ids,32);positions={sid:i for i,sid in enumerate(base.ids)}
 legacy=read(c['data']['input_pipeline']['legacy_seal']);conn=sqlite3.connect(Path(legacy['index']).as_uri()+'?mode=ro',uri=True)
 candidates=[x[0] for x in conn.execute('select sid from frames order by sid')];conn.close();rng.shuffle(candidates)
 ids.extend([x for x in candidates if x in positions and x not in ids][:12])
 indexed=sorted(range(len(base)),key=lambda i:base.lengths[i],reverse=True)
 ids.extend([base.ids[i] for i in indexed if base.ids[i] not in ids][:6])
 collator=PlanningCollator(c['model'],c['max_length'],c['max_pixels']);collator.get_processor();results=[];batch_a=[];batch_b=[]
 for j,sid in enumerate(ids):
  i=positions[sid]
  if j%2==0:
   t=time.monotonic();b=base[i];bs=time.monotonic()-t;t=time.monotonic();f=fast[i];fs=time.monotonic()-t
  else:
   t=time.monotonic();f=fast[i];fs=time.monotonic()-t;t=time.monotonic();b=base[i];bs=time.monotonic()-t
  assert b['target']==f['target'] and b['duration']==f['duration'] and b['split']==f['split']
  assert np.array_equal(b['audio'],f['audio']),sid+' audio'
  assert np.array_equal(np.asarray(b['image']),np.asarray(f['image'])),sid+' image'
  assert all(f['audit'][k]==v for k,v in b['audit'].items()),sid+' audit'
  batch_a.append(b);batch_b.append(f)
  if len(batch_a)==4 or j==len(ids)-1:
   original=collator(batch_a);improved=collator(batch_b)
   assert original.keys()==improved.keys()
   for k in original:assert torch.equal(original[k],improved[k]),sid+' tensor '+k
   batch_a=[];batch_b=[]
  result=dict(sample_id=sid,original_seconds=bs,fast_seconds=fs,route=f['audit']['first_frame_read_path'],equal=True)
  results.append(result);print(json.dumps(result),flush=True)
 report=dict(success=True,count=len(results),pipeline=c['data']['input_pipeline'],code={name:sha(ROOT/'src/p2_planning_sft'/name) for name in ('fast_media.py','reference_data.py','isolated_decode.py')},samples=results,
             original_seconds=sum(r['original_seconds'] for r in results),fast_seconds=sum(r['fast_seconds'] for r in results),processor_tensors_equal=True)
 atomic_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k not in ('samples','code','pipeline')}),flush=True)

if __name__=='__main__':main()
