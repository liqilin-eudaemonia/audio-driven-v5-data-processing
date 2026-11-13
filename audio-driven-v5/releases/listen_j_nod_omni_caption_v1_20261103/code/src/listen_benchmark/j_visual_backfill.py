"""Bounded scheduling-only supplement to the frozen visual measurement feeder.

Keep all configured CPU workers supplied when the foreground is already fed.
Never change measurement inputs, rules, the frozen contract, or source order.
"""
import json,os,time
from pathlib import Path
from .j_annotation import read,write,locked


def refill(cfg):
    if not cfg.get('visual_quality_backfill_enabled'):return dict(status='disabled',published=0)
    work=Path(cfg['visual_quality_gate']['work']);omni=Path(read(work/'config.json')['omni_work'])
    capacity=sum(int(n.get('visual_quality_workers',0)) for n in cfg['nodes'].values())
    target=capacity+max(32,capacity//2)
    if not 1<=capacity<=384:raise ValueError('Unvalidated quality CPU capacity')
    with locked(work/'feed.lock'):
        state=read(work/'feed.json');quality_config=read(work/'config.json');q=read(work/'queues/measure/state.json')
        # Foreground gets priority, including entries appended after its last read.
        if state['offset']<(omni/'ready.jsonl').stat().st_size:return dict(status='foreground_pending',published=0)
        pending=set(q['inflight'])|set(q.get('deferred',{}))
        # Count actual unclaimed IDs, not published-minus-completed: the latter
        # includes harmless replay/seed duplicates and can falsely throttle work.
        with (work/'ready.jsonl').open() as f:
            f.seek(q['offset'])
            for index,line in enumerate(f):
                if index>=2048:return dict(status='queue_scan_bound',published=0)
                if not line.endswith('\n'):break
                jid=json.loads(line)['job_id']
                if jid not in q.get('failed',{}) and not (work/'measure/completed'/f'{jid}.json').exists():pending.add(jid)
        before=len(pending);budget=min(128,max(0,target-before));entries=[];scanned=0
        if budget and state['backfill_offset']<quality_config['priority_start_offset']:
            with (omni/'ready.jsonl').open() as f:
                f.seek(state['backfill_offset'])
                while len(entries)<budget and scanned<512 and f.tell()<quality_config['priority_start_offset']:
                    line=f.readline()
                    if not line or not line.endswith('\n'):break
                    state['backfill_offset']=f.tell();scanned+=1;item=json.loads(line);jid=item['job_id']
                    if jid in pending or jid in q.get('failed',{}) or (work/'measure/completed'/f'{jid}.json').exists():continue
                    if not (omni/'caption/completed'/f'{jid}.json').exists():continue
                    entries.append(item);pending.add(jid)
        if entries:
            with (work/'ready.jsonl').open('a') as f:
                for item in entries:f.write(json.dumps(item)+'\n')
                f.flush();os.fsync(f.fileno())
        state['published']+=len(entries);state['at']=time.time();write(work/'feed.json',state)
        result=dict(at=time.time(),status='refilled' if entries else 'sufficient_or_no_backfill',configured_workers=capacity,target_sources=target,pending_before=before,published=len(entries),backfill_scanned=scanned,backfill_offset=state['backfill_offset'])
        write(work/'backfill_refill.json',result);return result
