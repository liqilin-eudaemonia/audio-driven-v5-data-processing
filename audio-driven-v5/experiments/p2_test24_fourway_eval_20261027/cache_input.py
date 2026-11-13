import sys,runpy
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
import json
arm,rank=sys.argv[1],int(sys.argv[2]);exps=json.loads((OUT/'experiments.json').read_text());run=exps[arm]['run']
runpy.run_path(str(ROOT/'scripts/p2_eval_generate.py'))['bootstrap'](['--run',run])
from p2_listen.features import cache_parts,folder,identity
from p2_listen.cache import load_part
from p2_listen.contracts import read
c=exps[arm]['config'];obs=read(OUT/'observations.json')[rank:rank+1]
missing=[o for o in obs if load_part(folder(c,'input',o['sample_id']),identity(c,'input',o),missing_ok=True) is None]
if missing:cache_parts(c,missing,{},['input'],'cuda:0')
print('verified',arm,[o['sample_id'] for o in obs],flush=True)
