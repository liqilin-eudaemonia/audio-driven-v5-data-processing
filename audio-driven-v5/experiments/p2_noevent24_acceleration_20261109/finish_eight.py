"""Finish only eight-NIC end-to-end test after the explicit user request."""
from pathlib import Path
import sys,time,copy,concurrent.futures
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C
from listen_cluster import launch,distributed_commands
C.write(W/'EIGHT_DRIVER.json',C.record());out=C.path('noevent',True);cl=copy.deepcopy(C.read(out/'cluster.json'))
for attempt in range(90):
 with concurrent.futures.ThreadPoolExecutor(3)as pool:
  obs=list(pool.map(lambda n:C.remote(cl,n,"import json,subprocess;print(json.dumps(dict(apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())))"),cl['nodes']))
 if all(not r['apps']for r in obs):break
 time.sleep(5)
else:raise RuntimeError('Selected GPUs occupied; no foreign process killed')
assert C.read(out/'status.json')['step']==224
cl['master_port']=29939;cl['network']['NCCL_IB_HCA']='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'
C.write(W/'EIGHT_CLUSTER.json',cl);C.write(W/'STATE.json',dict(at=time.time(),stage='eight_only_warmup_then_two_matched_repeats',formal_step=224))
launch(cl,distributed_commands(cl,W/'benchmark_eight_only.py',['--checkpoint',C.read(W/'SAVED_BOUNDARY.json')['checkpoint'],'--label','eight_only']),W/'eight_only_training_logs')
C.write(W/'STATE.json',dict(at=time.time(),stage='eight_only_complete',formal_step=224))
