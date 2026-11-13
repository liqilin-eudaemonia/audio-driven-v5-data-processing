"""Transfer only never-claimed sources to an isolated node queue.

No source media scan, model restart, or cross-node PID recovery. The original
queue prefix and offsets remain byte-identical. Both branch locks serialize
the handoff with running claims; a transaction receipt permits crash recovery.
"""
import json
from pathlib import Path
import time

from .j_annotation import digest, locked, read, write


def split_journal(raw,states,destination_fraction=.5):
    if isinstance(destination_fraction,bool) or not 0<destination_fraction<1:
        raise ValueError('Destination fraction must be between zero and one')
    lines=raw.splitlines(keepends=True)
    if any(not x.endswith(b'\n') for x in lines):raise ValueError('Incomplete ready journal')
    offsets={0};cursor=0;items=[]
    for line in lines:
        entry=json.loads(line)
        job=entry['job_id'];count=entry['count']
        if not isinstance(job,str) or Path(job).name!=job or job in ('.','..'):raise ValueError('Unsafe job')
        if type(count) is not int or count<1:raise ValueError('Invalid clip count')
        items.append((cursor,line,entry));cursor+=len(line);offsets.add(cursor)
    if any(s['offset'] not in offsets for s in states):raise ValueError('Queue offset not at journal boundary')
    cut=max(s['offset'] for s in states)
    protected={e['job_id'] for pos,_,e in items if pos<cut}
    for s in states:protected.update(s.get('inflight',{}));protected.update(s.get('failed',{}))
    counts={}
    for pos,line,e in items:
        job=e['job_id']
        if job in counts and counts[job]!=e['count']:raise ValueError('Conflicting repeated source')
        counts[job]=e['count']
    eligible={k:v for k,v in counts.items() if k not in protected}
    totals=[0,0];assigned=set()
    if destination_fraction==.5:
        for job,count in sorted(eligible.items(),key=lambda kv:(-kv[1],kv[0])):
            node=0 if totals[0]<=totals[1] else 1
            totals[node]+=count
            if node==1:assigned.add(job)
    else:
        target=sum(eligible.values())*destination_fraction;moved=0
        for job,count in sorted(eligible.items(),key=lambda kv:(-kv[1],kv[0])):
            if abs(moved+count-target)<abs(moved-target):
                assigned.add(job);moved+=count
    if not assigned:raise ValueError('No unclaimed sources to transfer')
    left=b''.join(line for _,line,e in items if e['job_id'] not in assigned)
    right=b''.join(line for _,line,e in items if e['job_id'] in assigned)
    assert left[:cut]==raw[:cut]
    assert set(assigned).isdisjoint(protected)
    return left,right,{'cut_offset':cut,'moved_sources':sorted(assigned),'destination_fraction':destination_fraction,
        'moved_clips':sum(counts[k] for k in assigned),'source_clips':sum(counts.values()),
        'remaining_clips_in_original_scope':sum(v for k,v in counts.items() if k not in assigned),
        'source_jobs':len(counts),'remaining_jobs_in_original_scope':len(counts)-len(assigned)}


def transfer_unclaimed(source,destination,transaction,destination_fraction=.5):
    source=Path(source).resolve();destination=Path(destination).resolve();transaction=Path(transaction).resolve()
    if source==destination:raise ValueError('Distinct node work directories required')
    if not (source/'ingest_complete.json').exists():raise ValueError('Ingestion must be complete')
    transaction.mkdir(parents=True,exist_ok=True);destination.mkdir(parents=True,exist_ok=True)
    with locked(transaction/'partition.lock'),locked(source/'ingest.lock'),locked(source/'queues/vlm/state.lock'),locked(source/'queues/nod/state.lock'):
        receipt=transaction/'partition.json';raw=(source/'ready.jsonl').read_bytes()
        if receipt.exists():
            plan=read(receipt)
            if plan.get('destination_fraction',.5)!=destination_fraction:raise ValueError('Partition fraction changed')
            if plan['source']!=str(source) or plan['destination']!=str(destination):raise ValueError('Transaction owner mismatch')
            if digest((destination/'ready.jsonl').read_bytes())!=plan['destination_sha256']:raise ValueError('Destination changed')
            if digest(raw)==plan['original_sha256']:
                if any(read(source/f'queues/{b}/state.json')['offset']>plan['cut_offset'] for b in ['vlm','nod']):
                    raise ValueError('Uncommitted plan became stale; do not apply after new claims')
                tmp=source/'ready.partition.tmp';tmp.write_bytes((transaction/'source_after.jsonl').read_bytes());tmp.replace(source/'ready.jsonl')
            elif digest(raw)!=plan['source_sha256']:raise ValueError('Source journal changed')
            plan['state']='committed';write(receipt,plan);return plan
        if (destination/'ready.jsonl').exists() or (destination/'supervisor.json').exists():raise ValueError('Destination already initialized')
        states=[read(source/f'queues/{b}/state.json') for b in ['vlm','nod']]
        left,right,meta=split_journal(raw,states,destination_fraction)
        plan=dict(meta,state='prepared',source=str(source),destination=str(destination),at=time.time(),
            original_sha256=digest(raw),source_sha256=digest(left),destination_sha256=digest(right))
        (transaction/'original_ready.jsonl').write_bytes(raw)
        (transaction/'source_after.jsonl').write_bytes(left)
        (transaction/'destination_ready.jsonl').write_bytes(right)
        write(transaction/'original_queue_states.json',dict(zip(['vlm','nod'],states)))
        (destination/'ready.jsonl').write_bytes(right)
        write(receipt,plan)
        tmp=source/'ready.partition.tmp';tmp.write_bytes(left);tmp.replace(source/'ready.jsonl')
        plan['state']='committed';write(receipt,plan)
        return plan
