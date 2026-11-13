"""Explicit stable resume at the saved224 boundary, never the earlier110 migration."""
from pathlib import Path
import sys,time
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C
from listen_cluster import launch,distributed_commands
out=C.path('noevent',True);cl=C.read(out/'cluster.json');saved=C.read(W/'SAVED_BOUNDARY.json')
assert saved['step']==224 and Path(saved['checkpoint']).with_suffix('.complete.json').exists()
if(out/'STOP').exists():
 assert C.read(out/'STOP')['owner']=='authorized-noevent24-acceleration-20261109';(out/'STOP').unlink()
C.write(S/'noevent_LAUNCH_CONTROLLER.json',C.record());C.write(out/'CONTROLLER.json',C.record())
cmds=distributed_commands(cl,S/'worker.py',['--arm','noevent','--config',str(out/'config.json'),'--resume',saved['checkpoint']])
launch(cl,cmds,out/('fallback_nodes_'+str(int(time.time()))))
