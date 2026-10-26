#!/usr/bin/env python3
"""Verify every backed-up file without needing data, credentials, or ML packages."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'BACKUP_MANIFEST.json').read_text())
for relative,row in manifest['files'].items():
 path=ROOT/relative
 assert path.is_file() and not path.is_symlink(),relative
 assert path.stat().st_size==row['bytes'],relative
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],relative
expected=set(manifest['files']);actual={str(p.relative_to(ROOT)) for p in (ROOT/'snapshot').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert actual==expected,dict(unexpected=sorted(actual-expected),missing=sorted(expected-actual))
print(json.dumps(dict(status='passed',files=len(expected),bytes=sum(r['bytes'] for r in manifest['files'].values()))))
