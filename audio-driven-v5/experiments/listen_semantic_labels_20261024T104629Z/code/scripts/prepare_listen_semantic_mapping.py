"""Freeze original annotations and create text-only semantic classification jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.semantic_mapping import make_items, digest, policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True);p.add_argument('--output', required=True)
    p.add_argument('--items-per-job', type=int, default=1)
    a = p.parse_args();source=Path(a.run).resolve();out=Path(a.output).resolve()
    if out.exists() or source in out.parents or a.items_per_job < 1: raise ValueError('Use new independent output')
    raw={name:(source/name).read_bytes() for name in ['labels.jsonl','provenance.json','input_review.json']}
    labels=[json.loads(line) for line in raw['labels.jsonl'].decode().splitlines() if line.strip()]
    items=make_items(labels,json.loads(raw['provenance.json']))
    jobs=[]
    for i in range(0,len(items),a.items_per_job):
        payloads={r['item_id']:r['payload'] for r in items[i:i+a.items_per_job]}
        jobs.append(dict(job_id=f'semantic-{len(jobs):03d}',payloads=payloads,payload_sha256=digest(payloads)))
    (out/'input').mkdir(parents=True)
    for name,content in raw.items():(out/'input'/name).write_bytes(content)
    manifest=dict(source_run=str(source),created_at=datetime.now(timezone.utc).isoformat(),
                  input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
                  items=len(items),jobs=len(jobs),policy=policy())
    for name,value in [('items.json',items),('jobs.json',jobs),('manifest.json',manifest)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    for rel in ['src/listen_benchmark/__init__.py','src/listen_benchmark/seven_class.py','src/listen_benchmark/semantic_mapping.py',
                'scripts/run_listen_semantic_mapping.py','scripts/prepare_listen_semantic_mapping.py']:
        q=out/'code'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,q)
    print(json.dumps(dict(output=str(out),items=len(items),jobs=len(jobs))))


if __name__=='__main__':main()
