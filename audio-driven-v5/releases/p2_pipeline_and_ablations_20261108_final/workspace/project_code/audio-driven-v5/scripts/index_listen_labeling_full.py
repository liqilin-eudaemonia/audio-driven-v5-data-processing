#!/usr/bin/env python3
"""Stream authorized primary manifests into disjoint source-pair shards."""
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_labeling.full_source import compact_source
from p2_listen.contracts import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--shards',type=int,default=64);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'index.json').exists():raise ValueError('Existing completed index; do not overwrite')
 (out/'shards').mkdir(exist_ok=True)
 handles=[(out/'shards'/f'{i:03d}.jsonl').open('w',buffering=1024*1024) for i in range(a.shards)]
 exclusions=(out/'excluded.jsonl').open('w');counts=Counter();reasons=Counter();shards=Counter();hours=Counter();began=time.monotonic()
 for split in ['train','dev','test']:
  source=(Path(a.source)/f'{split}_primary_keep_candidate.jsonl').resolve()
  with source.open() as f:
   for number,line in enumerate(f,1):
    row=json.loads(line);counts['seen']+=1
    try:value=compact_source(row,source,number)
    except (ValueError,KeyError,TypeError) as exc:
     counts['excluded']+=1;reasons[str(exc)]+=1;exclusions.write(json.dumps(dict(sample_id=row.get('sample_id'),split=split,line=number,reason=str(exc)))+'\n')
    else:
     shard=int(hashlib.sha256(value['pair'].encode()).hexdigest()[:8],16)%a.shards
     handles[shard].write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n');shards[shard]+=1;counts['eligible']+=1;counts[split]+=1;hours[split]+=value['duration_s']/3600
    if counts['seen']%10000==0:
     atomic_json(out/'progress.json',dict(counts=counts,reasons=reasons,elapsed_s=time.monotonic()-began,complete=False))
 for f in handles:f.close()
 exclusions.close()
 atomic_json(out/'index.json',dict(counts=counts,reasons=reasons,shards=shards,hours=hours,source_index=str(Path(a.source).resolve()/'index.json'),elapsed_s=time.monotonic()-began,complete=True,training_admitted=False))
 print(json.dumps(dict(counts=counts,hours=hours,elapsed_s=time.monotonic()-began)),flush=True)
if __name__=='__main__':main()
