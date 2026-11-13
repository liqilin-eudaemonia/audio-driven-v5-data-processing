"""Independent, strict search configurations; existing experiments are immutable."""
from pathlib import Path
import copy,math
from p2_listen.contracts import ROOT,read,sha
from p2_ltx_all_lora.config import expected
from . import VERSION
WORK=ROOT/'experiments/p2_loss_search28_20261110'
CANDIDATES={'baseline':(.05,.025),'event001':(.01,.025),'event020':(.2,.025),'event100':(1.,.025),'feature00025':(.05,.0025),'feature010':(.05,.1),'feature050':(.05,.5)}
def make(name):
 e,f=CANDIDATES[name];c=expected(WORK/'runs'/name)
 c['loss_search']=dict(version=VERSION,candidate=name,world=28,schedule=str(WORK/'schedule.json'),schedule_sha256=sha(WORK/'schedule.json'))
 c['training'].update(global_batch=28,steps=10151)
 c['loss_weights']=dict(av=1.,event=e,video_feature=f,audio_feature=f)
 return c

def normalized(c):
 # Native constructors are frozen to historical coefficients/batch31. They
 # construct the same architecture/data, not the objective or search sampler.
 return expected(c['output'])

def validate(c):
 name=c.get('loss_search',{}).get('candidate')
 if name not in CANDIDATES or c!=make(name):raise ValueError('Search config changed outside approved candidate coefficients')
 return c

def source_identity():
 from p2_ltx_all_lora.config import source_identity as native
 d=native()
 for f in [*Path(__file__).parent.glob('*.py'),WORK/'schedule.json',WORK/'protocol.json']:
  d[str(f)]=sha(f)
 return d


def validate_cluster(c,cluster):
 validate(c)
 base=read(ROOT/'experiments/p2_ltx_all_lora_noevent28_batch28_20261109/cluster.json')
 base=copy.deepcopy(base);base.update(master_port=30110+list(CANDIDATES).index(c['loss_search']['candidate']),training_script=str(Path(__file__).parent/'worker.py'),training_config=str(Path(c['output'])/'config.json'))
 if cluster!=base:raise ValueError('Cluster allocation, network or worker entry changed')
 if [(n['name'],n['gpus'])for n in cluster['nodes']]!=[(n,list(range(8)))for n in ('node12','node8','node4')]+[('node11',list(range(4)))]:raise ValueError('Only selected28 training GPUs allowed')
 return cluster
