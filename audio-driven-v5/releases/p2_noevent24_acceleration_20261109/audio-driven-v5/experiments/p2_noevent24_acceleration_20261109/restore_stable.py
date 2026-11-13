"""Wait for owned experiments to exit, then strictly resume original224 training."""
from pathlib import Path
import sys,time,subprocess,os,concurrent.futures
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109';sys.path.insert(0,str(S))
import control as C
C.write(W/'RESTORE_CONTROLLER.json',C.record())
cl=C.read(C.path('noevent',True)/'cluster.json')
for attempt in range(60):
 with concurrent.futures.ThreadPoolExecutor(3)as pool:
  observations=list(pool.map(lambda n:C.remote(cl,n,"import json,subprocess;print(json.dumps(dict(node="+repr(n['name'])+",apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())))"),cl['nodes']))
 C.write(W/'PRE_RESTORE_GPU.json',dict(at=time.time(),nodes=observations))
 if all(not n['apps']for n in observations):break
 time.sleep(5)
else:raise RuntimeError('GPUs still occupied; no foreign process will be killed')
C.write(W/'STATE.json',dict(at=time.time(),stage='restoring_stable_224',acceleration_adopted=False))
os.execv(C.PY,[C.PY,'-u',str(W/'fallback.py')])
