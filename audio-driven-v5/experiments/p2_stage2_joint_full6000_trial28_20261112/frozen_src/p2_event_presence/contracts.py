"""Explicit labels and immutable identities; omissions never mean absence."""
import hashlib
import json
import math
import os
from pathlib import Path

from . import SCHEMA

CLASSES = {
    'visual': ['nodding', 'head_shaking', 'smiling', 'frowning'],
    'audio': ['audible_laughter', 'brief_verbal_response', 'nonlexical_response'],
}
NAMES = {'nodding': '点头', 'head_shaking': '摇头', 'smiling': '微笑／笑态', 'frowning': '皱眉',
         'audible_laughter': '可听笑声', 'brief_verbal_response': '简短词语回应', 'nonlexical_response': '非词汇回应'}
STATES = {'present', 'absent', 'unknown', 'error'}
KNOWN = {'present', 'absent'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))
    temp.replace(path)


def labels(state='unknown', reason='not_observed_or_not_yet_evaluated'):
    if state not in STATES: raise ValueError('Invalid label state')
    return {mod: {name: {'state': state, 'evidence': reason} for name in names}
            for mod, names in CLASSES.items()}


def validate_labels(value):
    if not isinstance(value, dict) or set(value) != set(CLASSES): raise ValueError('Both modality maps required')
    for mod, names in CLASSES.items():
        if not isinstance(value[mod], dict) or set(value[mod]) != set(names):
            raise ValueError('Explicit complete class map required: ' + mod)
        for name in names:
            item = value[mod][name]
            if not isinstance(item, dict) or item.get('state') not in STATES:
                raise ValueError('Explicit categorical state required: ' + name)
            if not isinstance(item.get('evidence'), str) or not item['evidence'].strip():
                raise ValueError('Evidence/reason required: ' + name)
    return value


def collapse_events(events, duration_s, complete=True):
    """Independent event intervals may provide labels; their timing is never matched."""
    if not math.isfinite(duration_s) or duration_s <= 0: raise ValueError('Invalid duration')
    result = labels('absent' if complete else 'unknown', 'complete_event_scan' if complete else 'incomplete_event_scan')
    lookup = {c: mod for mod, names in CLASSES.items() for c in names}
    for event in events:
        c = event.get('class')
        if c not in lookup: raise ValueError('Unknown event class')
        a, b = event.get('start'), event.get('end')
        if (type(a) not in (float, int) or type(b) not in (float, int) or
                not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration_s):
            raise ValueError('Event must lie inside its OWN media interval')
        result[lookup[c]][c] = {'state': 'present', 'evidence': 'one_or_more_observed_instances'}
    return result


def pair_key(row):
    return tuple(row[k] for k in ('dataset', 'id', 'method', 'draw_id'))


def validate_record(row):
    if row.get('schema') != SCHEMA: raise ValueError('Unknown event schema')
    for key in ('dataset', 'id', 'method', 'draw_id', 'reference_identity', 'protocol_identity'):
        if not isinstance(row.get(key), str) or not row[key]: raise ValueError('Missing string field: ' + key)
    for mod in CLASSES:
        if type(row.get('capabilities', {}).get(mod)) is not bool: raise ValueError('Explicit modality capability required')
    validate_labels(row['reference']); validate_labels(row['generated'])
    return row
