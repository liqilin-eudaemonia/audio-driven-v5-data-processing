"""Parallel metadata-only manifest export; QC evidence stays in source results."""
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time


def shard(task):
    result_root,names,destination=task
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    handles={};counts=collections.Counter()
    try:
        for name in names:
            path=Path(result_root)/(name+'.json');value=json.loads(path.read_text())
            for row in value['clips']:
                key='_'.join((row['split'],row['admission_pool'],row['audio_final']['pool']))
                assert '/' not in key and '..' not in key
                if key not in handles:handles[key]=(destination/(key+'.jsonl')).open('w')
                compact={k:v for k,v in row.items() if k not in ('qc3','qc4')}
                compact['quality_evidence_file']=str(path)
                handles[key].write(json.dumps(compact,ensure_ascii=False)+'\n');counts[key]+=1
    finally:
        for h in handles.values():h.close()
    return dict(counts)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def export(out,summary,workers=16,shards=32):
    out=Path(out);manifest=out/'manifests';manifest.mkdir(exist_ok=True)
    names=sorted(p.stem for p in (out/'summaries').glob('*.json'))
    assert len(names)==summary['complete_pairs']
    temporary=manifest/('parallel_'+str(time.time_ns()));temporary.mkdir()
    tasks=[(str(out/'results'),names[i::shards],str(temporary/str(i))) for i in range(shards)]
    counts=collections.Counter();started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for value in pool.map(shard,tasks):counts.update(value)
    expected={k.replace('/','_'):v for k,v in summary['counts'].items()}
    assert dict(counts)==expected,(dict(counts),expected)
    hashes={}
    for key in sorted(counts):
        path=manifest/(key+'.jsonl.pending')
        with path.open('wb') as target:
            for i in range(shards):
                part=temporary/str(i)/(key+'.jsonl')
                if part.exists():
                    with part.open('rb') as source:shutil.copyfileobj(source,target,1024*1024)
        final=manifest/(key+'.jsonl');path.replace(final);hashes[key]=digest(final)
    index=dict(summary,manifest_sha256=hashes,export=dict(revision='parallel-identical-row-export-1',workers=workers,shards=shards,counts_verified=True,elapsed_seconds=time.monotonic()-started))
    pending=manifest/'index.json.pending';pending.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');pending.replace(manifest/'index.json')
    shutil.rmtree(temporary)
    return index
