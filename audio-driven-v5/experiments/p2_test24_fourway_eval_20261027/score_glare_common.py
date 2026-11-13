import sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,atomic_json
from p2_glare_eval.scoring import score_seed,aggregate_seeds
from p2_glare_eval.generation import check_idle_device
arm=sys.argv[1];device=sys.argv[2] if len(sys.argv)>2 else 'cuda:0'
if device.startswith('cuda'):check_idle_device(device)
keys={tuple(k) for k in read(OUT/'glare_common_scope.json')['keys']}
rows=[r for r in read(OUT/arm/'glare/detected_pairs.json') if (r['sample_id'],r['sample_seed']) in keys]
values={str(seed):score_seed([r for r in rows if r['sample_seed']==seed],device,32) for seed in [137,251]}
result=dict(arm=arm,device=device,evaluation_split='held_out_test',scope='four-arm common successfully detected clips, requiring both seeds; NOT full 24-clip result',pairs=len(rows),per_seed=values,aggregation=aggregate_seeds(values))
atomic_json(OUT/arm/'glare_common.json',result);print(arm,result['aggregation'],flush=True)
