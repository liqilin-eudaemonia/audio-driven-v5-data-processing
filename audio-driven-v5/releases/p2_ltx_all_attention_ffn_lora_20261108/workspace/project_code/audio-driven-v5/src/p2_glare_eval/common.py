import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f'Existing artifact identity differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def code_identity():
    files = sorted(Path(__file__).parent.glob('*.py')) + [ROOT / 'scripts/p2_glare_eval.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def checked_files(expected):
    for name, value in expected.items():
        if sha(name) != value:
            raise ValueError(f'File changed: {name}')


def job_key(sample_id, seed):
    return digest([sample_id, seed])[:32]


def artifact_valid(folder, identity):
    folder = Path(folder)
    marker = folder / 'complete.json'
    if not marker.exists():
        return False
    value = read(marker)
    if value['identity'] != digest(identity):
        raise ValueError(f'Artifact identity differs: {folder}; use a fresh work directory')
    for name, expected in value['files'].items():
        if sha(folder / name) != expected:
            raise ValueError(f'Artifact is corrupt: {folder / name}')
    return True


def complete(folder, identity, files, **metadata):
    folder = Path(folder)
    atomic_json(folder / 'complete.json', dict(identity=digest(identity), contract=identity,
        files={str(Path(p).relative_to(folder)): sha(p) for p in files}, **metadata))
