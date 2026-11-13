from pathlib import Path
import sys,json,time,copy,traceback
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import control as C
from listen_cluster import launch,distributed_commands
out=C.path('noevent',True)
def main():
 while C.alive(C.read(S/'noevent_LAUNCH_CONTROLLER.json')):time.sleep(3)
 step=C.read(out/'status.json')['step'];checkpoint=out/f'step_{step:06d}.pt';assert checkpoint.with_suffix('.complete.json').exists()
 C.write(W/'SAVED_BOUNDARY.json',dict(step=step,checkpoint=str(checkpoint),at=time.time()))
 for label,hca in [('single','=mlx5_0:1'),('eight','=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1')]:
  cfg=copy.deepcopy(C.read(out/'cluster.json'));cfg['master_port']=29931 if label=='single'else 29932;cfg['network']['NCCL_IB_HCA']=hca
  C.write(W/'STATE.json',dict(stage='benchmark',label=label,checkpoint=str(checkpoint),at=time.time()))
  args=['--checkpoint',str(checkpoint),'--label',label,'--candidate','combined']
  launch(cfg,distributed_commands(cfg,W/'benchmark.py',args),W/(label+'_training_logs'))
 C.write(W/'STATE.json',dict(stage='benchmarks_complete',at=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:C.write(W/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));raise
