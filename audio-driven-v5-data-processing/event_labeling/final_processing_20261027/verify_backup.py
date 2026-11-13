#!/usr/bin/env python3
"""Verify source snapshot and compressed memberships without GPU/model access."""
import gzip
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'BACKUP_MANIFEST.json').read_text())
for name,row in m['files'].items():
    p=ROOT/name
    assert p.is_file() and not p.is_symlink(),name
    raw=p.read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name
    if row.get('encoding')=='gzip':assert hashlib.sha256(gzip.decompress(raw)).hexdigest()==row['source_sha256'],name
actual={str(p.relative_to(ROOT)) for d in ['snapshot','membership','production_full_snapshot','production_stage_snapshots']
        for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
assert actual==set(m['files']),dict(extra=sorted(actual-set(m['files'])),missing=sorted(set(m['files'])-actual))
for b in json.loads((ROOT/'evidence/production_full_bindings.json').read_text()):
    assert m['files'][b['backup']]['sha256']==b['sha256']
for b in json.loads((ROOT/'evidence/production_stage_bindings.json').read_text()):
    assert m['files'][b['backup']]['sha256']==b['sha256']
print(json.dumps(dict(status='passed',files=len(actual),bytes=sum(r['bytes'] for r in m['files'].values()),frozen_python=m['production_frozen_python_files'])))
