#!/usr/bin/env python3
"""Verify archived code, live files, and external detector/model fingerprints."""
import hashlib
import json
from pathlib import Path

RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((RELEASE / 'manifest.json').read_text())
    failures = []
    for name, expected in manifest['files'].items():
        for path in (ROOT / name, RELEASE / 'code' / name):
            if not path.is_file() or sha256(path) != expected:
                failures.append(str(path))
    for name, expected in manifest['external_files'].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(str(path))
    print(json.dumps({'ok': not failures, 'files': len(manifest['files']),
                      'external_files': len(manifest['external_files']), 'failures': failures}, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
