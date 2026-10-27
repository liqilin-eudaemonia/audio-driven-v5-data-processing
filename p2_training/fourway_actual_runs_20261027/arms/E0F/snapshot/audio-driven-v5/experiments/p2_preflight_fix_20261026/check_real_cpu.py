import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'releases/p2_preflight_v1/src'))
from p2_listen.config import load,activate
from p2_listen.preflight import prepare_training
from p2_listen.dataset import load_observations,load_targets
from p2_listen.training_data import load_training_sample
from p2_listen.annotations import class_statistics
from p2_listen.contracts import read,digest,atomic_json
c=load(OUT/'fast/config.json');activate(c)
rows=load_observations(c,'train');targets=load_targets(c,rows);labels=[];signatures=[]
for o in rows:
    sample=load_training_sample(c,o,targets[o['sample_id']]);labels.append(sample[4]);del sample
    signatures.append(digest(read(Path(c['output'])/'labels/canonical'/(o['sample_id']+'.json'))))
stats=class_statistics(labels,c['events']['pos_weight_cap'],c['events']['K_train'])
a=prepare_training(c);b=prepare_training(c)
assert a==b and a['rows']==rows and a['targets']==targets and a['statistics']==stats and a['signatures']==signatures
atomic_json(OUT/'real_cpu_audit.json',dict(status='passed',samples=len(rows),statistics_exact=True,signatures_exact=True,rows_targets_exact=True,full_reuse_exact=True))
print('Real six-sample legacy/full/reuse metadata and statistics exactly match')
