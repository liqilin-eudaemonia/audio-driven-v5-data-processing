#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

def main():
    here = Path(__file__).resolve().parent
    m = json.loads((here / 'manifest.json').read_text())
    for rel, exp in m['archive_sha256'].items():
        p = here / 'code' / rel
        assert p.is_file(), f'Missing {p}'
        act = hashlib.sha256(p.read_bytes()).hexdigest()
        assert act == exp, f'SHA mismatch for {rel}: {act} != {exp}'
    print(f"{m['version']} verified: {len(m['archive_sha256'])} archived snapshot files OK")

if __name__ == '__main__':
    main()
