from pathlib import Path
import sys,time,copy,traceback,subprocess
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C
from listen_cluster import launch,distributed_commands
C.write(W/'TRIAL_CONTROLLER.json',C.record());out=C.path('noevent',True)
try:
 old=C.read(out/'CONTROLLER.json');assert C.alive(old)and not(out/'STOP').exists()
 C.write(W/'STATE.json',dict(stage='saving_old_boundary',at=time.time()))
 C.write(out/'STOP',dict(owner='batch24-speed-20261109',at=time.time(),reason='Authorized batch24 throughput experiment'))
 for _ in range(240):
  status=C.read(out/'status.json');cp=out/f"step_{status['step']:06d}.pt"
  if not C.alive(old)and cp.with_suffix('.complete.json').exists():break
  time.sleep(5)
 else:raise TimeoutError('Waiting complete boundary; do not kill training')
 C.write(W/'BOUNDARY.json',dict(step=status['step'],checkpoint=str(cp),sha256=C.read(cp.with_suffix('.complete.json'))['sha256'],parent=str(out),at=time.time()))
 import prepare
 prepare.prepare(status['step'])
 subprocess.run([C.PY,'-m','pytest','-q',str(W/'test_plan.py')],cwd=ROOT,check=True)
 cl=copy.deepcopy(C.read(C.read(out/'ACTIVE_CLUSTER.json')['path']));cl['master_port']=29945
 C.write(W/'BENCHMARK_CLUSTER.json',cl);C.write(W/'STATE.json',dict(stage='same744_throughput_test',at=time.time(),boundary=status['step']))
 launch(cl,distributed_commands(cl,W/'benchmark.py',['--checkpoint',str(cp),'--label','matched744']),W/'benchmark_logs')
 C.write(W/'STATE.json',dict(stage='benchmark_complete',at=time.time(),boundary=status['step']))
except BaseException as e:
 C.write(W/'ERROR.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
