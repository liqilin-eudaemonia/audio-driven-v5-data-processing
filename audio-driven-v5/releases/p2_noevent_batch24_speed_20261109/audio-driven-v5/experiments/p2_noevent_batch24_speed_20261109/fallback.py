from pathlib import Path
import sys,time
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C
from listen_cluster import launch,distributed_commands
b=C.read(W/'BOUNDARY.json');out=Path(b['parent']);cl=C.read(C.read(out/'ACTIVE_CLUSTER.json')['path']);cl['master_port']=29949
assert C.read(Path(b['checkpoint']).with_suffix('.complete.json'))['sha256']==b['sha256']
if(out/'STOP').exists():
 assert C.read(out/'STOP')['owner']=='batch24-speed-20261109';(out/'STOP').unlink()
C.write(out/'CONTROLLER.json',C.record());C.write(S/'noevent_LAUNCH_CONTROLLER.json',C.record())
launch(cl,distributed_commands(cl,S/'worker.py',['--arm','noevent','--config',str(out/'config.json'),'--resume',b['checkpoint']]),out/('batch24_fallback_'+str(int(time.time()))))
