import sys,json,os
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'releases/p2_feature_v1/src'))
from p2_listen.config import load,activate
from p2_listen.preflight import prepare_training
from p2_listen.contracts import atomic_json,digest
c=load(OUT/'production/config.json');activate(c)
result=prepare_training(c)
if int(os.environ.get('RANK','0'))==0:
 atomic_json(OUT/'production/audit_summary.json',{k:v for k,v in result.items() if k not in ('rows','targets','canonical')})
 print('FULL_DATASET_AUDIT_OK '+str(len(result['rows'])),flush=True)
