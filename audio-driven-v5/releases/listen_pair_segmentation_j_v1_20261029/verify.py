import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;r=p.parents[1];m=json.loads((p/'manifest.json').read_text())
for rel,h in m['code_sha256'].items():
 for f in [r/rel,p/'code'/rel]:
  assert hashlib.sha256(f.read_bytes()).hexdigest()==h,str(f)
print(m['version'], 'verified', len(m['code_sha256']), 'source and archive files')
