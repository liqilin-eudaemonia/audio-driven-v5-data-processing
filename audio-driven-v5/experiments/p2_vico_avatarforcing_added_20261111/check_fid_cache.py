from pathlib import Path
import sys,json,time,runpy
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(OUT)]
from vico_more_common import gpu_guard,save,sha
resource=gpu_guard(16)
import torch,numpy as np
from piq import FID
from fid_cache import install
torch.cuda.set_per_process_memory_fraction(.06,0)
src=ROOT/'evaluation/vico_challenge_baseline/evaluations/compute_fid.py'
read=runpy.run_path(str(src))['read_mp4']
protocol=json.loads((OUT/'image_eval/protocol.json').read_text());spec=protocol['datasets']['vico']
import pandas as pd
uid=pd.read_csv(spec['annotation']).iloc[0].uuid
paths=[Path(spec['gt_dir'])/(uid+'.listener.mp4'),OUT/'image_eval/scoring/avatarforcing'/(uid+'.listener.mp4')]
inputs=[torch.from_numpy(read(str(p),True,False,True)[:64]).float()/255 for p in paths]
def run(metric):
    torch.cuda.synchronize();start=time.monotonic()
    result=[metric.compute_feats([{'images':x}]) for x in inputs]
    torch.cuda.synchronize();return result,time.monotonic()-start
native,t1=run(FID());install();cached,t2=run(FID())
errors=[float((a-b).abs().max()) for a,b in zip(native,cached)]
assert max(errors)==0,errors
save(OUT/'FID_CACHE_PARITY.json',dict(passed=True,feature_max_abs=errors,samples_per_video=64,input_sha256={str(p):sha(p) for p in paths},original_seconds=t1,cached_seconds=t2,scope='Two batches only; whole-run acceleration not benchmarked',source_sha256=sha(OUT/'fid_cache.py'),official_script_sha256=sha(src),resource=resource))
print('Feature cache exact parity',errors,flush=True)
