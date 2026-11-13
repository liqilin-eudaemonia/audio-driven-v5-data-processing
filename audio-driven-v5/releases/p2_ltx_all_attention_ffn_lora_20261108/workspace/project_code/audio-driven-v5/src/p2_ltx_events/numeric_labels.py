"""CPU-only compilation of existing clip labels; no annotation policy changes.

Contiguous arrays replace JSON decoding, category matching and per-class tensor
creation in the training loop. The sparse bank remains the source of truth.
"""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from p2_listen.contracts import read, sha, digest, atomic_json
from p2_clip_events.labels import LabelReader
from p2_clip_events.model import classes

SCHEMA = 'p2-clip-numeric-v1'
ARRAY_DTYPES = dict(y='float32', known='bool', weight='float32', source_digests='S64')


def validate_guard(manifest, names):
    guard=manifest.get('evidence_guard')
    if guard is not None:
        from .evidence_guard import POLICY
        if (guard['policy']!=POLICY or guard['classes']!=list(names)
                or guard['code_sha256']!=sha(Path(__file__).with_name('evidence_guard.py'))):
            raise ValueError('Guarded label rules/classes changed; rebuild guarded labels')


def contract(labels, names):
    m = labels.manifest
    validate_guard(m,names)
    return dict(label_manifest_sha256=sha(labels.folder/'manifest.json'),
                label_database_sha256=m['database_sha256'],
                classes=list(names), absence_weight=m['absence_weight'],
                label_policy=m['label_policy'],
                tensor_rules_sha256=sha(Path(__file__).parents[1]/'p2_clip_events/labels.py'))


def prepare_numeric(c, output):
    """Publish an immutable cache only after all rows and source hashes validate."""
    if not c.get('current_cache_index'):
        raise ValueError('Numeric preparation requires current-cache configuration')
    labels = LabelReader(c['clip_labels'])
    temporary = None
    try:
        names = classes(c['event_supervision'])
        if not names or len(set(names)) != len(names):
            raise ValueError('Expected unique nonempty ordered classes')
        source = read(Path(c['current_source'])/'manifest.json')
        ids = labels.ids()
        if (ids != source['training_ids'] or len(ids) != labels.manifest['samples']
                or labels.manifest['cache_manifest_sha256'] != sha(Path(c['current_source'])/'manifest.json')):
            raise ValueError('Numeric label source/order mismatch')
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Expected unique nonempty sample IDs')
        weight = labels.manifest['absence_weight']
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Invalid absence weight')
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError('Use a new numeric label output directory')
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=output.name+'.tmp-', dir=output.parent))
        identity = contract(labels, names)
        shape = (len(ids), len(names))
        arrays = {key: np.lib.format.open_memmap(temporary/(key+'.npy'), mode='w+',
                  dtype=dtype, shape=(len(ids),) if key=='source_digests' else shape)
                  for key, dtype in ARRAY_DTYPES.items()}
        count = 0
        for ordinal, sid, value in labels.conn.execute('SELECT ordinal,sid,value FROM labels ORDER BY ordinal'):
            row = json.loads(value)
            if ordinal != count or sid != ids[count] or row['sample_id'] != sid:
                raise ValueError('Numeric label row/order mismatch')
            source_digest = row['source_digest']
            if (len(source_digest) != 64 or any(ch not in '0123456789abcdef' for ch in source_digest)
                    or source_digest != source['entries'][sid]):
                raise ValueError('Stale clip target: '+sid)
            positive, unknown = set(row['positive']), set(row['unknown'])
            # Exact p2_clip_events.labels.tensors semantics, including the stored
            # weight on masked entries; effective weight is known * weight.
            present = np.asarray([name in positive for name in names], dtype=bool)
            known = np.asarray([name not in unknown for name in names], dtype=bool)
            known &= present | (row['allow_weak'] and weight > 0)
            arrays['y'][count] = present
            arrays['known'][count] = known
            arrays['weight'][count] = np.where(present, 1., weight)
            arrays['source_digests'][count] = source_digest
            count += 1
        if count != len(ids):
            raise ValueError('Incomplete numeric label rows')
        # Check again before publishing, so a concurrent source change fails closed.
        if (identity != contract(labels, names)
                or sha(labels.folder/'labels.sqlite') != identity['label_database_sha256']
                or sha(Path(c['current_source'])/'manifest.json') != labels.manifest['cache_manifest_sha256']):
            raise ValueError('Label sources changed during preparation')
        stats = {name: dict(positive=int(((arrays['y'][:, j]==1)&arrays['known'][:, j]).sum()),
                           weak_negative=int(((arrays['y'][:, j]==0)&arrays['known'][:, j]).sum()),
                           masked=int((~arrays['known'][:, j]).sum())) for j,name in enumerate(names)}
        for value in arrays.values():
            value.flush()
        arrays.clear()
        atomic_json(temporary/'sample_ids.json', ids)
        paths = ['sample_ids.json']+[key+'.npy' for key in ARRAY_DTYPES]
        manifest = dict(schema=SCHEMA, contract=identity, samples=count, shape=list(shape),
                        sample_ids_sha256=digest(ids), statistics=stats,
                        files={name: sha(temporary/name) for name in paths},
                        source_split_counts=source.get('source_split_counts'),
                        semantics='identical_to_sparse_tensors; masked_weights_not_zeroed',
                        evidence_conflict_audit='not_performed_by_numeric_compiler')
        atomic_json(temporary/'manifest.json', manifest)
        # Read back all serialized shapes, dtypes and hashes before publication.
        NumericLabels(temporary, labels, names, ids)
        os.rename(temporary, output)
        temporary = None
        return dict(output=str(output), samples=count, classes=len(names),
                    config_entry={'clip_numeric_labels': str(output)}, training_started=False)
    finally:
        labels.conn.close()
        if temporary is not None:
            shutil.rmtree(temporary)


class NumericLabels:
    """Shared CPU memory maps; callers receive independent small row tensors."""
    def __init__(self, folder, labels, names, ids, *, verify=True):
        self.folder = Path(folder)
        self.manifest = read(self.folder/'manifest.json')
        m = self.manifest
        self.names = tuple(names)
        self.ids = read(self.folder/'sample_ids.json')
        expected_files = {'sample_ids.json'} | {key+'.npy' for key in ARRAY_DTYPES}
        if (m['schema'] != SCHEMA or m['contract'] != contract(labels, names)
                or self.ids != list(ids) or len(set(self.ids)) != len(self.ids)
                or m['sample_ids_sha256'] != digest(self.ids)
                or m['samples'] != len(ids) or m['shape'] != [len(ids), len(names)]
                or set(m['files']) != expected_files):
            raise ValueError('Numeric label identity/classes/order mismatch')
        if verify:
            for name, expected in m['files'].items():
                if sha(self.folder/name) != expected:
                    raise ValueError('Numeric label checksum mismatch: '+name)
        self.arrays = {key: np.load(self.folder/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                       for key in ARRAY_DTYPES}
        for key, dtype in ARRAY_DTYPES.items():
            expected_shape = (len(ids),) if key=='source_digests' else (len(ids), len(names))
            if self.arrays[key].shape != expected_shape or self.arrays[key].dtype != np.dtype(dtype):
                raise ValueError('Numeric label shape/dtype mismatch: '+key)

    def events(self, index, sid, source_digest, device='cpu'):
        if not 0 <= index < len(self.ids) or self.ids[index] != sid:
            raise ValueError('Numeric label sample/index mismatch')
        if self.arrays['source_digests'][index].decode('ascii') != source_digest:
            raise ValueError('Stale numeric clip target: '+sid)
        # Three contiguous row copies/transfers instead of three tiny allocations
        # per class. Copies keep read-only shared maps safe from caller mutation.
        values = {key: torch.from_numpy(np.array(self.arrays[key][index], copy=True)).to(device)
                  for key in ('y', 'known', 'weight')}
        return {'clip/'+name: {key: value[j:j+1] for key, value in values.items()}
                for j, name in enumerate(self.names)}
