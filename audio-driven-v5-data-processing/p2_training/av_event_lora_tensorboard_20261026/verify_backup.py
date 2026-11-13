#!/usr/bin/env python3
"""Verify every snapshot file without importing ML packages or accessing GPUs."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'BACKUP_MANIFEST.json').read_text())
assert manifest['count'] == len(manifest['files'])
for item in manifest['files']:
    path = root / item['path']
    assert path.resolve().is_relative_to(root), item['path']
    assert path.is_file() and not path.is_symlink(), item['path']
    assert path.stat().st_size == item['bytes'], item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
print('Verified', manifest['count'], 'snapshot files; bytes:', manifest['bytes'])
