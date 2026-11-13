"""Lossless, content-verified cache around the sealed Seamless GT decoder.

The official reader, scorer and their code/feature-cache identities stay intact.
This optional execution wrapper records its own identity in each score contract.
"""
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def archive_stamp(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


class DecodedGTCache:
    def __init__(self, root, reader=None):
        if reader is None:
            from evaluation.seamless import reader
        self.reader = reader
        self.original = reader.decode_reference
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        versions = {}
        for name in ('av', 'numpy', 'Pillow'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.identity = dict(schema='seamless-decoded-rgb-cache-v1',
                             wrapper_sha256=file_sha(__file__),
                             reader_sha256=file_sha(reader.__file__), versions=versions)
        self.events = []

    def _identity(self, reference):
        # Re-run the existing frozen-row, crop and upstream-reader validation on
        # every hit. resolve also verifies caption/quality source metadata.
        row, resolved = self.reader.resolve(reference)
        archive = resolved['source']['source']['archive']
        return dict(implementation=self.identity, reference=reference,
                    resolved_source=resolved['source'],
                    crop_provenance=resolved['crop_provenance'],
                    archive=archive_stamp(archive))

    def _read(self, folder, identity):
        meta = json.loads((folder/'metadata.json').read_text())
        if meta['identity'] != identity:
            raise ValueError('Decoded GT cache identity mismatch')
        frames = np.load(folder/'frames.npy', mmap_mode='r', allow_pickle=False)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (512, 512, 3):
            raise ValueError('Decoded GT cache RGB geometry mismatch')
        if list(frames.shape) != meta['shape'] or len(frames) != meta['audit']['frames']:
            raise ValueError('Decoded GT cache frame count mismatch')
        if hashlib.sha256(memoryview(frames)).hexdigest() != meta['audit']['rgb_sha256']:
            raise ValueError('Decoded GT cache pixel checksum mismatch')
        if archive_stamp(identity['archive']['path']) != identity['archive']:
            raise ValueError('Seamless source archive changed during cache read')
        return frames, meta['fps'], copy.deepcopy(meta['audit'])

    def get(self, reference):
        start = time.perf_counter()
        identity = self._identity(reference)
        key = digest(identity)
        shard = self.root/key[:2]
        shard.mkdir(exist_ok=True)
        folder = shard/key
        hit = folder.is_dir()
        # Atomic directory publication means readers of a committed entry need
        # no exclusive lock. Multiple score workers can mmap the same GT.
        if hit:
            result = self._read(folder, identity)
        else:
            with (shard/(key+'.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                hit = folder.is_dir()
                if not hit:
                    temp = Path(tempfile.mkdtemp(prefix=key+'.', dir=shard))
                    try:
                        frames, fps, audit = self.original(reference)
                        if frames.dtype != np.uint8 or frames.shape[1:] != (512, 512, 3):
                            raise ValueError('Original GT decoder returned unexpected RGB')
                        if hashlib.sha256(memoryview(np.ascontiguousarray(frames))).hexdigest() != audit['rgb_sha256']:
                            raise ValueError('Original GT decoder RGB audit mismatch')
                        if archive_stamp(identity['archive']['path']) != identity['archive']:
                            raise ValueError('Seamless source archive changed during decoding')
                        np.save(temp/'frames.npy', frames, allow_pickle=False)
                        meta = dict(identity=identity, fps=fps, audit=audit, shape=list(frames.shape))
                        (temp/'metadata.json').write_text(json.dumps(meta, allow_nan=False)+'\n')
                        os.replace(temp, folder)
                    finally:
                        if temp.exists():
                            shutil.rmtree(temp)
                result = self._read(folder, identity)
        self.events.append(dict(id=reference['sample_id'], hit=hit,
                                seconds=time.perf_counter()-start, key=key,
                                rgb_sha256=result[2]['rgb_sha256']))
        return result

    @contextlib.contextmanager
    def installed(self):
        previous = self.reader.decode_reference
        self.reader.decode_reference = self.get
        try:
            yield self
        finally:
            self.reader.decode_reference = previous

    def report(self):
        return dict(identity=self.identity, root=str(self.root),
                    hits=sum(e['hit'] for e in self.events),
                    misses=sum(not e['hit'] for e in self.events), events=self.events,
                    original_reader_unchanged=True, pixel_checksum_checked_each_read=True)
