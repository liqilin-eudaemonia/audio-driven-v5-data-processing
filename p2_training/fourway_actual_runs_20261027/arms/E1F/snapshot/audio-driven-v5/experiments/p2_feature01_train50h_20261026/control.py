"""Frozen AV+Feature launch and baseline E0 comparability verification."""
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

BASELINE=ROOT/f'experiments/p2_preflight_fix_20261026/production/runs/{SEED}/E0'
def verify_config(c,old):
 a=dict(c);b=dict(old);a.pop('output');b.pop('output')
 ta=dict(a.pop('training'));tb=dict(b.pop('training'))
 expected={'feature_loss_enabled':True,'feature_lr':3e-5,'feature_loss_weights':{'video':.1,'audio':.1}}
 for k,v in expected.items():
  if ta.pop(k,None)!=v:raise ValueError('Unexpected Feature training setting: '+k)
  tb.pop(k,None)
 if a!=b or ta!=tb:raise ValueError('Unexpected configuration difference from E0')
 if Path(c['output']).resolve()!=TRAIN:raise ValueError('Unexpected output')
def verify_release():
 release=read(OUT/'release.json')
 if (release['arm'],release['seed'],release['steps'])!=('E0',SEED,BUDGET):raise ValueError('Unexpected experiment')
 for path,expected in release['files'].items():
  if sha(path)!=expected:raise ValueError('Frozen launch input changed: '+path)
 verify_config(read(TRAIN/'config.json'),read(BASELINE/'run.json')['config'])
 return release
def verify_pair():
 a=read(BASELINE/'run.json');b=read(E0/'run.json');release=verify_release()
 allowed=('config','source','feature_loss_enabled','initialization','shared_initialization')
 differences=[k for k in set(a)|set(b) if k not in allowed and a.get(k)!=b.get(k)]
 if differences:raise ValueError('Pair differs: '+str(differences))
 if b['shared_initialization']!=a['initialization']:raise ValueError('Shared initialization differs from E0')
 if b['source']!=release['source']:raise ValueError('Unapproved training source')
 if not b['feature_loss_enabled']:raise ValueError('Feature branch disabled')
 verify_config(b['config'],a['config'])
 return dict(status='passed',baseline=str(BASELINE),allowed_differences=list(allowed),
  shared_initialization=b['shared_initialization'],observations=a['observations'],targets=a['targets'],
  seed=SEED,budget=BUDGET,world=b['world'],checked_at=now(),event_gradient_to_generator=False)
