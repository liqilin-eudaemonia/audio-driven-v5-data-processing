import sys,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,DEFAULTS,subprocess_env
from p2_omni_eval.backends import Backends
from p2_omni_eval import media
runtime=read(ROOT/'configs/p2_omni_runtime.lock.json');os.environ.update(subprocess_env(runtime,'cpu'))
p=dict(runtime=runtime,options=DEFAULTS);backend=Backends(p,'cpu')
rows=read(ROOT/'experiments/p2_preflight_fix_20261026/data/manifests/targets.json');dest=OUT/'model_smoke_v2';dest.mkdir(exist_ok=True)
results={}
for i,t in enumerate(rows[:2]):
 sub=dest/str(i);sub.mkdir(exist_ok=True);g=media.geometry(t['target_video']);a=media.audio(t['target_audio'],sub/'listener.wav',g['duration_s'])
 for metric,fn in [('fvd',lambda:backend.fvd_features(t['target_video'],media.window(t['sample_id'],g['frame_count'],DEFAULTS),sub/'i3d.npy')),
                   ('utmos',lambda:backend.mos(a)),('sync',lambda:backend.sync(t['target_video'],a,sub/'sync'))]:
  begin=time.time()
  try:result=fn()
  except Exception as e:result=dict(status='error',error=repr(e))
  result['seconds']=time.time()-begin;results[f'{i}_{metric}']=result;atomic_json(OUT/'model_smoke_v2.json',results);print(metric,result,flush=True)
try:results['text']=backend.text('yes that sounds good','yes that sounds good')
except Exception as e:results['text']=dict(status='error',error=repr(e))
atomic_json(OUT/'model_smoke_v2.json',results);print(results['text'],flush=True)
