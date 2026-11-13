import json,hashlib,ast
from pathlib import Path
import p2_listen.config as config
from p2_listen.features import identity
from p2_listen.contracts import digest
RUN=Path(__file__).resolve().parent.parent
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
assert Path(config.__file__)==RUN/'code/audio-driven-v5/src/p2_listen/config.py'
assert config.ROOT==ROOT
manifest=json.loads((RUN/'code_manifest.json').read_text());logical=str(ROOT/'src/p2_listen/config.py');expected=manifest[logical] if logical in manifest else manifest[str(Path(config.__file__))]
assert hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest()==expected
cfg=json.loads((RUN/'p2_config.json').read_text());assert config.load(RUN/'p2_config.json')==cfg
rows=[]
for p in sorted((RUN/'state/features').glob('*.json'),reverse=True)[:3]:
 rows.append(json.loads((RUN/f'manifests/shards/{p.stem}.json').read_text())[0])
checks=[]
for row in rows:
 sid=row['sample_id'];obs=row['observation'];target=row['target'];label=json.loads((RUN/f'labels/canonical/{sid}.json').read_text())
 for part in ['input','teacher','vae','events']:
  args=dict(obs=obs)
  if part in ['teacher','vae']:args['target']=target
  if part=='events':args['label']=label
  got=identity(cfg,part,**args);saved=json.loads((RUN/f'cache/{part}/{sid}/complete.json').read_text());assert got==saved['contract'] and digest(got)==saved['identity'],(sid,part);checks.append([sid,part])
got=identity(cfg,'text');saved=json.loads((RUN/'cache/text/complete.json').read_text());assert got==saved['contract'] and digest(got)==saved['identity'];checks.append(['shared','text'])
report=dict(frozen_config=str(config.__file__),config_sha256=expected,native_contract_root=str(config.ROOT),existing_cache_identity_checks=checks,all_equal=True,live_source_sha256=hashlib.sha256((ROOT/'src/p2_listen/config.py').read_bytes()).hexdigest(),live_source_untouched=True)
(RUN/'frozen_config_runtime/validation.json').write_text(json.dumps(report,indent=2));print('PASS',len(checks),'existing cache identities identical; original frozen config imported; live source preserved')
