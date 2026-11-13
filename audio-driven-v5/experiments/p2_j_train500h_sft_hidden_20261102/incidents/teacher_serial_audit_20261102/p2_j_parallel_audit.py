#!/usr/bin/env python3
"""Bounded parallel execution of the unchanged per-sample teacher audit checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,read,digest,sha
_reader=None


def check_process(index,ids):return check_block(_reader,index,ids)


def check_block(reader,index,ids):
    work=reader.work
    receipt=read(work/'done'/f'{index:06d}.json')
    if not receipt['success'] or receipt['sample_ids']!=ids or receipt['protocol_sha256']!=reader.protocol_sha:
        raise ValueError('Wrong task completion receipt')
    if len(receipt['receipts'])!=len(ids) or {r['sample_id'] for r in receipt['receipts']}!=set(ids):
        raise ValueError('Invalid per-sample receipts')
    total=0
    for r in receipt['receipts']:
        sid=r['sample_id'];folder=work/'cache'/sid;marker=read(folder/'complete.json')
        expected=reader.identity(reader.item(sid))
        if (marker['contract']!=expected or marker['identity']!=digest(expected)
                or marker['payload_sha256']!=r['payload_sha256'] or marker['identity']!=r['identity']
                or (folder/'payload.pt').stat().st_size!=r['bytes']):
            raise ValueError('Invalid teacher cache: '+sid)
        total+=r['bytes']
    return len(ids),total


def main():
    global _reader
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--processes',type=int,default=0)
    a=p.parse_args()
    from p2_j_teacher.core import Reader
    reader=Reader(a.work);work=reader.work;jobs=read(work/'jobs.json');seen=set();start=time.time()
    if list((work/'failed').glob('*.json')):raise ValueError('Extraction has unresolved failures')
    for ids in jobs:
        if seen & set(ids) or len(set(ids))!=len(ids):raise ValueError('Duplicate completed members')
        seen.update(ids)
    if seen!=reader.ids:raise ValueError('Teacher membership differs')
    count=total_bytes=blocks=0
    _reader=reader
    executor=(ProcessPoolExecutor(max_workers=a.processes,mp_context=multiprocessing.get_context('fork')) if a.processes
              else ThreadPoolExecutor(max_workers=a.threads))
    with executor as pool:
        futures=([pool.submit(check_process,i,ids) for i,ids in enumerate(jobs)] if a.processes else
                 [pool.submit(check_block,reader,i,ids) for i,ids in enumerate(jobs)])
        for f in as_completed(futures):
            n,b=f.result();count+=n;total_bytes+=b;blocks+=1
            if blocks%16==0 or blocks==len(jobs):
                atomic_json(work/'parallel_audit_progress.json',dict(samples=count,total=len(seen),blocks=blocks,
                            seconds=time.time()-start,threads=0 if a.processes else a.threads,processes=a.processes,time=time.time()))
    result=dict(success=True,samples=len(seen),payload_bytes=total_bytes,finished_at=time.time(),
        protocol_sha256=reader.protocol_sha,source=reader.protocol['source'],
        writer_validation='per-payload SHA256 and exact tensor roundtrip',
        final_validation='complete member coverage, source identity, per-sample receipt/hash binding and byte size')
    atomic_json(work/'parallel_audit_result.json',dict(**result,seconds=time.time()-start,threads=0 if a.processes else a.threads,processes=a.processes,audit_code_sha256=sha(__file__)))
    if (work/'COMPLETE.json').exists():
        old=read(work/'COMPLETE.json')
        for key in ('success','samples','payload_bytes','protocol_sha256','source'):
            if old[key]!=result[key]:raise ValueError('Independent audit mismatch: '+key)
    else:atomic_json(work/'COMPLETE.json',result,immutable=True)
    print(dict(success=True,samples=count,seconds=time.time()-start),flush=True)


if __name__=='__main__':main()
