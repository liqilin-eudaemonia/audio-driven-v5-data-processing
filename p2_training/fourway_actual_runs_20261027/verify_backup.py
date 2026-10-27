"""Verify portable snapshot bytes against the actual four training source contracts."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(B/'BACKUP_MANIFEST.json')
for item in m['files']:
 p=B/item['path'];assert p.is_file() and sha(p)==item['sha256'],str(p)
counts={}
for e in read(B/'EXPERIMENTS.json'):
 arm=e['arm'];d=B/'arms'/arm/'evidence';run=read(d/'run.json');bindings=read(d/'source_bindings.json')
 assert len(bindings)==len(run['source'])==e['source_count']
 for item in bindings:
  assert run['source'][item['original']]==item['sha256']==sha(B/item['backup'])
 assert read(d/'config.json')==run['config']
 assert read(d/'step_001268.complete.json')['sha256']==e['checkpoint_sha256']
 counts[arm]=len(bindings)
print(json.dumps(dict(status='passed',files=len(m['files']),actual_training_source_bindings=counts,weights_included=False)))
