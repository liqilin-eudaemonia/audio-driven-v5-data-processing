"""Frozen E1+Feature launch paired against the existing E0+Feature run."""
import hashlib,json,os,tempfile
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
REFERENCE=ROOT/'experiments/p2_feature01_train50h_20261026'
TRAIN=REFERENCE/'production'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';SEED=20261023;BUDGET=1268
E1=TRAIN/f'runs/{SEED}/E1'
BASELINE=TRAIN/f'runs/{SEED}/E0'
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

def verify_release():
 release=read(OUT/'release.json')
 if (release['arm'],release['seed'],release['steps'])!=('E1',SEED,BUDGET):raise ValueError('Unexpected E1+Feature experiment')
 for path,expected in release['files'].items():
  if sha(path)!=expected:raise ValueError('Frozen launch input changed: '+path)
 c=read(OUT/'config.json');prior=read(BASELINE/'run.json')
 if c!=prior['config'] or c!=read(TRAIN/'config.json'):raise ValueError('E1+Feature configuration differs from E0+Feature')
 if release['source']!=prior['source']:raise ValueError('Source differs from paired E0+Feature')
 if not c['training']['feature_loss_enabled'] or c['training']['feature_loss_weights']!={'video':.1,'audio':.1} or c['events']['weight']!=.05:
  raise ValueError('Unexpected loss configuration')
 return release

def validate_resume(path):
 if path is None:return
 path=Path(path).resolve()
 if path.parent!=E1.resolve():raise ValueError('Resume must belong to this E1+Feature run; E0 checkpoints are not a fresh E1 initialization')
 if not path.is_file() or not path.with_suffix('.complete.json').is_file():raise ValueError('Resume checkpoint or completion marker missing')

def compare_pair(a,b):
 if a['arm']!='E0' or b['arm']!='E1':raise ValueError('Expected E0+Feature / E1+Feature arms')
 differences=[k for k in set(a)|set(b) if k!='arm' and a.get(k)!=b.get(k)]
 if differences:raise ValueError('Pair differs beyond arm: '+str(differences))
 return dict(status='passed',baseline=str(BASELINE),experiment=str(E1),allowed_differences=['arm'],
  initialization=b['initialization'],shared_initialization=b['shared_initialization'],
  seed=SEED,budget=BUDGET,world=b['world'],checked_at=now(),event_gradient_to_generator=True)

def verify_pair():
 verify_release()
 return compare_pair(read(BASELINE/'run.json'),read(E1/'run.json'))
