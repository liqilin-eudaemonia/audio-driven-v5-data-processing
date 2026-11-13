"""Freeze reference labels once per clip; compare candidates against identical GT.

The first baseline evaluation supplies reference labels. The 200-clip baseline
may extend the registry after the 64-clip stage, but never relabel existing IDs.
Unknown/error states are retained, so downstream comparisons must check coverage.
"""
from pathlib import Path
import copy
import fcntl
import json
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from p2_listen.contracts import read, sha, atomic_json
from p2_event_presence.contracts import digest, validate_record, rows, save_rows
from p2_event_presence.scoring import score


def align(records, registry, expected_ids, allow_extend=False):
    """Pure validation/merge; fresh generated labels are never replaced."""
    records = [validate_record(r) for r in records]
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Event population differs from frozen validation IDs')
    if len({(r['dataset'], r['method'], r['draw_id']) for r in records}) != 1:
        raise ValueError('One dataset/method/draw required per evaluation')
    protocols = {r['protocol_identity'] for r in records}
    if len(protocols) != 1:
        raise ValueError('Mixed detector protocols')
    merged = copy.deepcopy(registry)
    if merged and {v['protocol_identity'] for v in merged.values()} != protocols:
        raise ValueError('Frozen reference detector protocol changed')
    fixed = []
    for row in records:
        sid = row['id']
        if sid not in merged:
            if not allow_extend:
                raise ValueError('Reference must be established by baseline first: ' + sid)
            merged[sid] = {k: copy.deepcopy(row[k]) for k in
                          ('id', 'dataset', 'reference_identity', 'protocol_identity', 'reference')}
        reference = merged[sid]
        for key in ('dataset', 'reference_identity', 'protocol_identity'):
            if row[key] != reference[key]:
                raise ValueError('Reference media or protocol changed: ' + sid)
        fixed.append(dict(row, reference=copy.deepcopy(reference['reference'])))
    return fixed, merged


def freeze(work):
    work = Path(work)
    p = read(work / 'protocol.json')
    name = p['checkpoint_identity']['experiment']['loss_search']['candidate']
    count = p['total_samples']
    if count not in (64, 200):
        raise ValueError('Unapproved validation size')
    manifest = WORK / f'tuning{count}.json'
    membership = read(WORK / 'VALIDATION_MEMBERSHIP_ACCEPTED.json')
    if sha(manifest) != membership['outputs'][manifest.name]:
        raise ValueError('Validation membership changed')
    expected = [r['id'] for r in read(manifest)]
    records = rows(work / 'events/event_pairs.jsonl')
    arm = p['arms'][0]['name']
    if any(r['dataset'] != 'seamless' or r['method'] != arm for r in records):
        raise ValueError('Events belong to another evaluation')
    frozen = WORK / 'reference_events'
    frozen.mkdir(exist_ok=True)
    with (frozen / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = {}
        for path in sorted(frozen.glob('*.json')):
            item = read(path)
            if path.stem != digest(item['id']):
                raise ValueError('Reference registry key changed')
            registry[item['id']] = item
        fixed, merged = align(records, registry, expected, allow_extend=name == 'baseline')
        for sid, item in merged.items():
            atomic_json(frozen / (digest(sid) + '.json'), item, immutable=True)
        refs = {sid: sha(frozen / (digest(sid) + '.json')) for sid in expected}
    save_rows(work / 'event_pairs_fixed_reference.jsonl', fixed)
    expected_pairs = rows(work / 'events/frozen_pairs.jsonl')
    atomic_json(work / 'fixed_reference_event_summary.json', score(fixed, expected_pairs))
    receipt = dict(success=True, samples=count, candidate=name,
                   reference_sha256_by_id=refs,
                   raw_events_sha256=sha(work / 'events/event_pairs.jsonl'),
                   fixed_events_sha256=sha(work / 'event_pairs_fixed_reference.jsonl'),
                   summary_sha256=sha(work / 'fixed_reference_event_summary.json'),
                   unknown_is_negative=False, detector_based=True,
                   human_calibrated=False)
    atomic_json(work / 'REFERENCE_EVENTS_ACCEPTED.json', receipt)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().work.resolve())))
