"""Stop only the disposable exploratory job after its exact candidates are checked."""
from pathlib import Path
import sys,time,json,os,signal,copy,subprocess
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C
from listen_cluster import launch,distributed_commands
old=C.read(W/'DRIVER.json')
while True:
 p=W/'single/RESULTS.json'
 if p.exists()and len(C.read(p)['results'])>=4:break
 if not C.alive(old):raise RuntimeError('Exploratory job stopped before exact candidates completed')
 time.sleep(2)
assert 'run_bench.py'in Path('/proc',str(old['pid']),'cmdline').read_bytes().decode()
os.kill(old['pid'],signal.SIGTERM)
# Closing the driver's guard pipes stops only its experimental process groups.
time.sleep(40)
C.write(W/'EXPLORATORY_STOP.json',dict(at=time.time(),driver=old,reason='Fused numerics rejected; exact reuse and reporting candidates retained'))
out=C.path('noevent',True);saved=C.read(W/'SAVED_BOUNDARY.json')
base=C.read(out/'cluster.json')
for attempt in range(120):
 empty=True
 for node in base['nodes']:
  check=C.remote(base,node,"import subprocess,json;print(json.dumps(dict(apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())))")
  empty &= not check['apps']
 if empty:break
 time.sleep(5)
else:raise RuntimeError('Experimental GPUs not empty; do not overlap full models')

for label,hca in [('single_exact','=mlx5_0:1'),('eight_exact','=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1')]:
 cfg=copy.deepcopy(C.read(out/'cluster.json'));cfg['master_port']=29935 if label=='single_exact'else 29936;cfg['network']['NCCL_IB_HCA']=hca
 C.write(W/'STATE.json',dict(stage='matched_exact_abba',label=label,at=time.time()))
 launch(cfg,distributed_commands(cfg,W/'benchmark_exact.py',['--checkpoint',saved['checkpoint'],'--label',label,'--candidate','combined']),W/(label+'_training_logs'))
C.write(W/'STATE.json',dict(stage='exact_benchmarks_complete',at=time.time()))
