import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,atomic_json
from p2_glare_eval.pipeline import prepare_pairs,evaluate
rows=[dict(sample_id=t['sample_id'],sample_seed=137,gt_video=t['target_video'],generated_video=t['target_video']) for t in read(OUT/'targets.json')]
atomic_json(OUT/'gt_pairs.json',dict(schema='p2-glare-pairs-v1',samples=rows),immutable=True)
prepare_pairs(OUT/'gt_pairs.json',OUT/'gt_detection')
r=evaluate(OUT/'gt_detection',workers=12,detection_only=True);print(r['status'],r['valid_pairs'],r['failed_pairs'],flush=True)
