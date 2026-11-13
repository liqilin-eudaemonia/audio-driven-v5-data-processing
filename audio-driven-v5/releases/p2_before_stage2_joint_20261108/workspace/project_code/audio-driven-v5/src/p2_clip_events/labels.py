"""Sparse clip labels derived from existing, evidence-backed open statistics."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from itertools import islice
import multiprocessing
from pathlib import Path
import sqlite3
import torch

from p2_listen.contracts import read, digest, sha, atomic_json
from . import VERSION


def select_classes(manifest, min_positive):
    """Select a train vocabulary in frozen order, without filtering any rows."""
    if type(min_positive) is not int or min_positive < 1:
        raise ValueError('min_positive must be a positive integer')
    groups = {mod: [name for name in names
                    if manifest['statistics'][name]['positive'] >= min_positive]
              for mod, names in manifest['classes'].items()}
    if not any(groups.values()):
        raise ValueError('No clip classes meet min_positive')
    return groups


def derive(item):
    s = item['statistics']
    if s['sample_id'] != item['sample_id'] or s['caption'] != item['caption']:
        raise ValueError('Statistics/caption identity mismatch')
    positive = set(s['counted_types'])
    uncertain = set()
    for e in s['evidence_ledger']:
        raw = e['raw']
        key = ('visual' if raw['modality'] == 'visual' else 'audio') + ':' + e['canonical']
        # Negated wording may describe a subinterval (e.g. "no additional speech").
        # Do not promote it to a verified whole-clip negative.
        if e['disposition'] not in ('counted', 'baseline'):
            uncertain.add(key)
    conflict = positive & uncertain
    positive -= conflict
    allow_weak = (s['record_status'] == 'computed' and not s['pending_behavior_count']
                  and not s['unusable_behaviors'] and not s['original_validation_error'])
    return dict(sample_id=item['sample_id'], positive=sorted(positive), unknown=sorted(uncertain),
                allow_weak=allow_weak, caption_sha256=item['caption_sha256'], source_digest=digest(item),
                statistics_version=s['statistics_version'], conflicts=sorted(conflict))


def tensors(row, names, absent_weight, device='cpu'):
    positive, unknown = set(row['positive']), set(row['unknown'])
    result = {}
    for name in names:
        present = name in positive
        known = name not in unknown and (present or row['allow_weak'] and absent_weight > 0)
        result[name] = dict(y=torch.tensor([float(present)], device=device),
                            known=torch.tensor([known], device=device),
                            weight=torch.tensor([1. if present else absent_weight], device=device))
    return result


def _derive_file(args):
    path, expected = args
    item = read(path)
    if digest(item) != expected:
        raise ValueError('Frozen source changed: ' + str(path))
    if item['split'] != 'train':
        raise ValueError('Only train labels may define the training vocabulary')
    return derive(item)


def prepare(work, output, *, processes=16, sample_ids=None, absent_weight=.1):
    from p2_j_cache.data import open_manifest
    if not 0 <= absent_weight <= 1:
        raise ValueError('Absent-label weight must be in [0, 1]')
    work, output = Path(work).resolve(), Path(output).resolve()
    m = open_manifest(work)
    ids = list(m['splits']['train'] if sample_ids is None else sample_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(m['splits']['train']):
        raise ValueError('Expected unique selected train members')
    output.mkdir(parents=True, exist_ok=True)
    db = output / 'labels.sqlite'
    if db.exists() or (output / 'manifest.json').exists():
        raise FileExistsError('Use a new clip-label output directory')
    pos, weak_pos, weak_unknown = Counter(), Counter(), Counter()
    weak_rows = conflicts = count = 0
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE labels (ordinal INTEGER PRIMARY KEY, sid TEXT UNIQUE NOT NULL, value TEXT NOT NULL)')
        with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context('fork')) as pool:
            args = ((work/'samples'/(sid+'.json'), m['entries'][sid]) for sid in ids)
            # Bounded submission also works on Python 3.13 (no map buffersize).
            while batch := list(islice(args, 4096)):
                for row in pool.map(_derive_file, batch, chunksize=32):
                    count += 1
                    conn.execute('INSERT INTO labels VALUES (?,?,?)', (count-1, row['sample_id'], json.dumps(row)))
                    pos.update(row['positive']);conflicts += len(row['conflicts'])
                    if row['allow_weak']:
                        weak_rows += 1;weak_pos.update(row['positive']);weak_unknown.update(row['unknown'])
                conn.commit()
                print(json.dumps(dict(stage='clip_labels', samples=count, total=len(ids))), flush=True)
        conn.commit()
    names = sorted(pos)
    if not names:
        raise ValueError('No positive behavior labels')
    groups = dict(video=[k for k in names if k.startswith('visual:')],
                  audio=[k for k in names if k.startswith('audio:')])
    if len(groups['video']) + len(groups['audio']) != len(names):
        raise ValueError('Unknown behavior modality')
    stats = {}
    for k in names:
        negative = weak_rows - weak_pos[k] - weak_unknown[k]
        enabled = pos[k] > 0 and negative > 0 and absent_weight > 0
        stats[k] = dict(positive=pos[k], weak_negative=negative,
                        known_weight=pos[k] + absent_weight*negative, enabled=enabled,
                        pos_weight=max(1., min(10., absent_weight*negative/pos[k])),
                        disabled_reason=None if enabled else 'no_positive_negative_contrast')
    result = dict(schema=VERSION, cache_work=str(work), cache_manifest_sha256=sha(work/'manifest.json'),
                  database_sha256=sha(db), samples=count, sample_ids_sha256=digest(ids),
                  full_train=ids == m['splits']['train'], classes=groups, statistics=stats,
                  absence_weight=absent_weight, label_policy='counted_positive; uncertain_mask; omitted_weak_negative',
                  temporal_targets=False, class_mapping='existing_open_names_without_new_merges',
                  conflicts_masked=conflicts, human_ground_truth=False)
    atomic_json(output/'manifest.json', result, immutable=True)
    return result


class LabelReader:
    def __init__(self, folder, *, verify=True):
        self.folder = Path(folder)
        self.manifest = read(self.folder/'manifest.json')
        if self.manifest['schema'] != VERSION:
            raise ValueError('Wrong clip label version')
        if verify and sha(self.folder/'labels.sqlite') != self.manifest['database_sha256']:
            raise ValueError('Clip label database changed')
        self.conn = sqlite3.connect((self.folder/'labels.sqlite').resolve().as_uri()+'?mode=ro', uri=True)

    def row(self, sid):
        row = self.conn.execute('SELECT value FROM labels WHERE sid=?', (sid,)).fetchone()
        if row is None:
            raise KeyError(sid)
        return json.loads(row[0])

    def ids(self):
        return [r[0] for r in self.conn.execute('SELECT sid FROM labels ORDER BY ordinal')]
