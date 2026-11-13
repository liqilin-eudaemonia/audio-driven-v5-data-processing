import json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/private/liqi_workspace/project_code/OmniResponse')
from evaluation.video.paired_metrics.backends import Networks
from evaluation.video.paired_metrics.media import load_pair
torch.set_num_threads(2)
w=Path(__file__).parent;rows=[json.loads(x) for x in (w/'benchmark_pairs.jsonl').read_text().splitlines()]
g,p,m=load_pair(rows[-1]);assets=json.loads(Path('/private/liqi_workspace/project_code/OmniResponse/evaluation/video/paired_metrics/assets.local.json').read_text())
model=Networks(assets,'cuda:0',64);values={};records=[]
# Warm both batch shapes, then ABBA to reduce drift from concurrent evaluation.
for batch in [64,128,64,128,128,64]:
 model.batch_size=batch;torch.cuda.synchronize();start=time.perf_counter();x=np.array(model.lpips(g,p));torch.cuda.synchronize();elapsed=time.perf_counter()-start
 if batch in values:np.testing.assert_allclose(x,values[batch],rtol=1e-6,atol=1e-7)
 values[batch]=x;records.append(dict(batch=batch,seconds=elapsed));print(records[-1],flush=True)
np.testing.assert_allclose(values[64],values[128],rtol=1e-5,atol=1e-6)
result=dict(frames=len(g),records=records,max_abs_difference=float(np.abs(values[64]-values[128]).max()),context='concurrent production evaluation; two warmups then ABBA',mean_seconds={str(b):float(np.mean([r['seconds'] for r in records[2:] if r['batch']==b])) for b in [64,128]})
(w/'BATCH64_VS128.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
