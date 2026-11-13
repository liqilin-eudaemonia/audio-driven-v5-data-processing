"""Shared GT-only feature cache; generated outputs are never cached here."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class GTFeatureCache:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def get(self, frames, kind, parameters, compute):
        pixels = np.ascontiguousarray(frames)
        identity = dict(version=1, backend=self.identity, kind=kind,
                        shape=list(pixels.shape), dtype=str(pixels.dtype),
                        rgb_sha256=hashlib.sha256(memoryview(pixels)).hexdigest(),
                        parameters=parameters)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.root/kind/key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/(key+'.npz')
        with (folder/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    result = tuple(data[f'arr_{i}'].copy() for i in range(len(data.files)))
                return result, True
            result = tuple(np.asarray(x) for x in compute())
            fd, temporary = tempfile.mkstemp(prefix=key+'.', suffix='.tmp', dir=folder)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    np.savez(stream, *result)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return result, False
