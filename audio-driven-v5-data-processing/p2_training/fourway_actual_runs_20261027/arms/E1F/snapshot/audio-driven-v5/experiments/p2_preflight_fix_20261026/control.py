"""Frozen E0 launch inputs and strict cross-release E1/E0 pair verification."""
import hashlib,json,os,tempfile
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];TRAIN=OUT/'production'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';SEED=20261023;BUDGET=1268
E1=ROOT/f'experiments/p2_e1_train50h_lora_tb_20261026/runs/{SEED}/E1';E0=TRAIN/f'runs/{SEED}/E0'
HOSTS=('172.29.252.11','172.29.15.91','172.29.225.51')
HOSTNAMES=('dedicated-developjob-lq-dev-machine-1nkpf','dedicated-developjob-lq-dev-machine-2-taaft','dedicated-developjob-lq-dev-machine-3-zhw7q')
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def approved_source(old):
 base=ROOT/'src/p2_listen';release=ROOT/'releases/p2_preflight_v1'
 replacements={str(base/n):str(release/'src/p2_listen'/n) for n in ('__init__.py','training.py','config.py')}
 expected={replacements.get(p,p):sha(replacements.get(p,p)) for p in old}
 for p in (release/'src/p2_listen/preflight.py',release/'scripts/p2_listen.py',ROOT/'scripts/p2_listen.py'):expected[str(p)]=sha(p)
 if any(sha(p)!=h for p,h in old.items()):raise ValueError('Original E1 source changed')
 return expected
def verify_release():
 release=read(OUT/'release.json')
 if (release['arm'],release['seed'],release['steps'])!=('E0',SEED,BUDGET):raise ValueError('Unexpected successor')
 for path,expected in release['files'].items():
  if sha(path)!=expected:raise ValueError('Frozen launch input changed: '+path)
 c=read(TRAIN/'config.json');old=read(E1/'run.json')['config']
 if {k:v for k,v in c.items() if k!='output'}!={k:v for k,v in old.items() if k!='output'}:raise ValueError('Configuration differs beyond output')
 if Path(c['output']).resolve()!=TRAIN:raise ValueError('Unexpected output')
 approved_source(read(E1/'run.json')['source'])
 return release
def verify_pair():
 a=read(E1/'run.json');b=read(E0/'run.json')
 if a['arm']!='E1' or b['arm']!='E0':raise ValueError('Wrong arms')
 differences=[k for k in set(a)|set(b) if k not in ('arm','source','config') and a.get(k)!=b.get(k)]
 if differences:raise ValueError('Pair differs: '+str(differences))
 if {k:v for k,v in a['config'].items() if k!='output'}!={k:v for k,v in b['config'].items() if k!='output'}:raise ValueError('Configuration differs')
 if b['source']!=approved_source(a['source']):raise ValueError('Unapproved source delta')
 return dict(status='passed',allowed_differences=['arm','config.output','explicit_preflight_release_sources'],
  initialization=a['initialization'],observations=a['observations'],targets=a['targets'],labels=len(a['labels']),
  seed=SEED,budget=BUDGET,world=b['world'],checked_at=now(),original_E1_source_preserved=True)
