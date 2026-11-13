"""Read-only data audit; writes only its separate review receipt."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import p2_current500h_component_ablations as cli
import numpy as np
import torch
from p2_listen.contracts import read, sha, atomic_json
from p2_current_component_ablation.config import load, source_identity
from p2_current_component_ablation.data import Dataset, to_device
from p2_whisper_event_feature.execution import CPUInputStream
from p2_j_cache.data import caption_prompt

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent
started = time.time()
assert source_identity() == read(OUT/'SOURCE_IDENTITY.json')
report = dict(success=False, started=started, arms={}, training_source_unchanged=True)
for arm in cli.ARMS:
    c = load(OUT/arm/'config.json')
    d = Dataset(c)
    canaries = set(read(Path(c['j_cache'])/'protocol.json')['canary_ids'])
    rng = random.Random(20261108)
    chosen = []
    for split, count in [('train', 12), ('dev', 4)]:
        pool = [i for i, row in enumerate(d.rows)
                if row['split'] == split and row['sample_id'] not in canaries]
        chosen.extend(rng.sample(pool, count))
    requests = [(step, [(index, step)]) for step, index in enumerate(chosen, 1)]
    records = []
    with CPUInputStream(d, requests, 2) as stream:
        for step, item in enumerate(stream, 1):
            assert item['step'] == step and item['chosen'] == requests[step-1][1]
            value = to_device(item['samples'][0], 'cpu')
            sid = value['sample_id']
            row = d.reader.source.row(sid)
            assert value['text']['positive']['prompt'] == caption_prompt(row['caption'])
            assert (value.get('teacher') is not None) == cli.ARMS[arm]['feature']
            assert (value.get('whisper') is not None) == cli.ARMS[arm]['whisper']
            assert ('events' in value) == cli.ARMS[arm]['event']
            records.append(dict(sample_id=sid, split=row['split'], caption_sha256=row['caption_file_sha256']))
    report['arms'][arm] = dict(success=True, random_prefetch_reads=records,
                              config_sha256=sha(OUT/arm/'config.json'))
    print(json.dumps(dict(arm=arm, random_prefetch_reads=len(records), success=True)), flush=True)
    # Both Event arms use exactly the same frozen label and numeric caches.
    if arm == 'no_whisper':
        source, labels, numeric = d.main_aux['event']
        arrays = numeric.arrays
        names = numeric.names
        absent = labels.manifest['absence_weight']
        guarded = unknown = weak = positive = count = 0
        for ordinal, sid, raw in labels.conn.execute('SELECT ordinal,sid,value FROM labels ORDER BY ordinal'):
            row = json.loads(raw)
            assert ordinal == count and sid == d.ids[count] == row['sample_id']
            assert arrays['source_digests'][count].decode() == row['source_digest'] == source.manifest['entries'][sid]
            pos, unk = set(row['positive']), set(row['unknown'])
            y = np.asarray([n in pos for n in names], dtype=np.float32)
            known = np.asarray([n not in unk and (n in pos or row['allow_weak'] and absent > 0)
                                for n in names], dtype=bool)
            weight = np.asarray([1. if n in pos else absent for n in names], dtype=np.float32)
            assert np.array_equal(y, arrays['y'][count])
            assert np.array_equal(known, arrays['known'][count])
            assert np.array_equal(weight, arrays['weight'][count])
            guard = row.get('evidence_guard', {}).get('masked_classes', [])
            assert not set(guard) & pos and set(guard) <= unk
            guarded += len(guard)
            unknown += int((~known).sum())
            weak += int((known & (y == 0)).sum())
            positive += int((known & (y == 1)).sum())
            count += 1
            if count % 100000 == 0:
                print(json.dumps(dict(numeric_rows_verified=count)), flush=True)
        assert count == len(d.rows) == 284218
        report['full_numeric_labels'] = dict(success=True, rows=count, classes=len(names),
            class_targets=count*len(names), positive=positive, weak_negative=weak,
            masked=unknown, guarded_mask_entries=guarded,
            note='Before the unchanged minimum-positive class enable mask; not a new visual annotation audit')
        print(json.dumps(report['full_numeric_labels']), flush=True)
    del d
report.update(success=True, finished=time.time(), cuda_initialized=torch.cuda.is_initialized(),
              gpu_acceptance='pending allocation; no GPU training started by this review')
assert not report['cuda_initialized']
atomic_json(OUT/'REVIEW_DATA_ACCEPTED.json', report)
print(json.dumps(dict(success=True, seconds=report['finished']-started)), flush=True)
