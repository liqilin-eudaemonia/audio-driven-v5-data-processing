"""One-time locked transfer of unclaimed sources into the existing local queue."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.j_annotation import read, write, locked
from listen_benchmark.j_queue_partition import transfer_unclaimed

BASE = Path(__file__).resolve().parent
SCOPE = BASE.parent / 'nodes.json'


def entries(raw):
    result = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if row['job_id'] in result and result[row['job_id']] != row['count']:
            raise ValueError('Conflicting source counts')
        result[row['job_id']] = row['count']
    return result


def append_staged(destination, stage, transaction):
    with locked(destination / 'ingest.lock'), locked(destination / 'queues/vlm/state.lock'), locked(destination / 'queues/nod/state.lock'):
        raw = (destination / 'ready.jsonl').read_bytes()
        extra = (stage / 'ready.jsonl').read_bytes()
        current = entries(raw); added = entries(extra)
        overlap = set(current) & set(added)
        receipt = transaction / 'append.json'
        if overlap:
            if overlap != set(added) or any(current[k] != v for k, v in added.items()):
                raise ValueError('Partial or inconsistent destination transfer')
            if not receipt.exists():
                raise ValueError('Unowned destination sources')
            plan = read(receipt)
        else:
            # Node12 uses the original complete jobs catalog, so transferred jobs
            # require no copying and no changes to source media or existing labels.
            if any(not (destination / 'jobs' / (job + '.json')).exists() for job in added):
                raise ValueError('Destination job catalog incomplete')
            plan = {'state': 'prepared', 'at': time.time(), 'destination_prefix_bytes': len(raw),
                'destination_prefix_sha256': hashlib.sha256(raw).hexdigest(),
                'added_sources': len(added), 'added_clips': sum(added.values())}
            (transaction / 'destination_before.jsonl').write_bytes(raw)
            write(receipt, plan)
            temp = destination / 'ready.rebalance.tmp'
            temp.write_bytes(raw + extra)
            temp.replace(destination / 'ready.jsonl')
        plan['state'] = 'committed'
        write(receipt, plan)
        return plan


def main():
    scope = read(SCOPE); destination = Path(scope['nodes']['node12']['work'])
    plan_file = BASE / 'balance_plan.json'
    if not plan_file.exists():
        remaining = {}; eligible = {}; union = {}
        for node, cfg in scope['nodes'].items():
            work = Path(cfg['work']); raw = (work / 'ready.jsonl').read_bytes(); rows = entries(raw)
            if union.keys() & rows.keys():
                raise ValueError('Existing source ownership overlaps')
            union.update(rows)
            states = [read(work / 'queues' / branch / 'state.json') for branch in ('vlm', 'nod')]
            remaining[node] = sum(rows.values()) - min(s['completed_clips'] for s in states)
            protected = set(entries(raw[:max(s['offset'] for s in states)]))
            for s in states:
                protected.update(s['inflight']); protected.update(s['failed'])
            eligible[node] = sum(count for job, count in rows.items() if job not in protected)
        assert len(union) == scope['total_sources'] and sum(union.values()) == scope['total_clips']
        target = sum(remaining.values()) / 3
        plan = {'at': time.time(), 'remaining_before': remaining, 'target_remaining': target,
            'fractions': {n: (remaining[n] - target) / eligible[n] for n in ('node17', 'node8')}}
        assert all(0 < f < 1 for f in plan['fractions'].values())
        write(BASE / 'source_counts_before.json', union)
        write(plan_file, plan)
    plan = read(plan_file)
    for node in ('node17', 'node8'):
        transaction = BASE / ('balance_from_' + node); stage = transaction / 'staging'
        transfer_unclaimed(scope['nodes'][node]['work'], stage, transaction, plan['fractions'][node])
        append_staged(destination, stage, transaction)
    union = {}
    for node, cfg in scope['nodes'].items():
        rows = entries((Path(cfg['work']) / 'ready.jsonl').read_bytes())
        assert not union.keys() & rows.keys()
        union.update(rows)
        cfg.update(scope_sources=len(rows), scope_clips=sum(rows.values()))
    assert union == read(BASE / 'source_counts_before.json')
    for path in (SCOPE, ROOT / 'data_corpora/seamless_listen_j_visual_quality_v1_20261030/annotation_nodes.json'):
        write(path, scope)
    report = {'passed': True, 'at': time.time(), 'sources_disjoint': True,
        'source_union_and_counts_unchanged': True, 'total_sources': len(union),
        'total_clips': sum(union.values()), 'nodes': scope['nodes']}
    write(BASE / 'balance_audit.json', report)
    print(json.dumps({n: {k: c[k] for k in ('scope_sources', 'scope_clips')} for n, c in scope['nodes'].items()}))


if __name__ == '__main__':
    main()
