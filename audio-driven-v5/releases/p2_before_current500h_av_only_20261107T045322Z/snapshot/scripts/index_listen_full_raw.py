"""Freeze exactly the two accepted manifests into reusable bounded shards."""
import argparse, hashlib, json, os, sqlite3, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import compact, views, VERSION
from p2_listen.contracts import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--run',required=True);p.add_argument('--shards',type=int,default=1024);a=p.parse_args()
    source=Path(a.index).resolve();run=Path(a.run).resolve();q=run/'queue';q.mkdir(parents=True,exist_ok=True)
    if (q/'index.json').exists() or (q/'shards').exists():raise ValueError('Use fresh index directory')
    (q/'shards').mkdir();spec=json.loads(source.read_text());(q/'accepted_index.json').write_bytes(source.read_bytes())
    if spec['split']!='train':raise ValueError('Unexpected index split')
    handles=[(q/'shards'/f'{i:04d}.jsonl').open('wb',buffering=1024*256) for i in range(a.shards)]
    db=sqlite3.connect(f'/tmp/listen_raw_index_{os.getpid()}.sqlite');db.execute('create table ids (id text primary key)')
    counts=Counter();shards=Counter();began=time.time();verified=[]
    with (q/'excluded.jsonl').open('w') as excluded:
        for part in spec['parts']:
            path=(source.parent/part['path']).resolve();digest=hashlib.sha256();seen=0
            with path.open('rb') as f:
                for number,line in enumerate(f,1):
                    digest.update(line);row=json.loads(line);seen+=1;counts['seen']+=1
                    db.execute('insert into ids values (?)',(row['sample_id'],))
                    try:r=compact(row,path,number,hashlib.sha256(line).hexdigest())
                    except (ValueError,TypeError,KeyError) as exc:
                        counts['source_invalid']+=1;excluded.write(json.dumps(dict(sample_id=row.get('sample_id'),line=number,manifest=str(path),error=str(exc)))+'\n')
                    else:
                        k=int(hashlib.sha256(r['pair'].encode()).hexdigest()[:8],16)%a.shards
                        handles[k].write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
                        counts['queued']+=1;counts[r['admission_pool']]+=1;counts['views_per_branch']+=len(views(r['duration_s']))
                        counts['crop_unavailable']+=r['video_transform'] is None;shards[k]+=1
                    if counts['seen']%10000==0:
                        db.commit();atomic_json(q/'progress.json',dict(counts=counts,seconds=time.time()-began,complete=False))
            if seen!=part['clips'] or digest.hexdigest()!=part['sha256']:raise ValueError('Source count/hash mismatch: '+str(path))
            verified.append(dict(path=str(path),sha256=digest.hexdigest(),count=seen))
    if counts['seen']!=spec['clips']:raise ValueError('Total count mismatch')
    for h in handles:h.close()
    db.close()
    atomic_json(q/'index.json',dict(version=VERSION,counts=counts,shards=shards,verified_sources=verified,
        accepted_index_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seconds=time.time()-began,complete=True))
    print(json.dumps(dict(counts=counts,seconds=time.time()-began)),flush=True)

if __name__=='__main__':main()
