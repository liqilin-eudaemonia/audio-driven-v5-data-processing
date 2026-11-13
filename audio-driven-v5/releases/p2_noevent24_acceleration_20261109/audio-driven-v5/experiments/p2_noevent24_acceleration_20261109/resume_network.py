"""Network-only adoption: unchanged training code, output, model, Adam, and RNG."""
from pathlib import Path
import sys,time
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(S),str(ROOT/'scripts'),str(ROOT/'src')]
import control as C,runtime
from listen_cluster import launch,distributed_commands
ready=C.read(W/'NETWORK_ADOPTION_READY.json');out=C.path('noevent',True);cl=C.read(W/'NETWORK_PRODUCTION_CLUSTER.json')
assert ready['success']and ready['cluster_sha256']==C.sha(W/'NETWORK_PRODUCTION_CLUSTER.json')
assert ready['config_sha256']==C.sha(out/'config.json')and ready['source']==runtime.source_identity('noevent')==C.read(out/'run.json')['source']
assert ready['eight_results_sha256']==C.sha(W/'eight_only/RESULTS.json')
assert C.read(out/'status.json')['step']==224
saved=C.read(W/'SAVED_BOUNDARY.json');assert saved['step']==224
assert C.read(Path(saved['checkpoint']).with_suffix('.complete.json'))['sha256']==ready['checkpoint_sha256']
if(out/'STOP').exists():
 assert C.read(out/'STOP')['owner']=='authorized-noevent24-acceleration-20261109';(out/'STOP').unlink()
record=C.record();C.write(S/'noevent_LAUNCH_CONTROLLER.json',record);C.write(out/'CONTROLLER.json',record);C.write(W/'PRODUCTION_CONTROLLER.json',record)
cmds=distributed_commands(cl,S/'worker.py',['--arm','noevent','--config',str(out/'config.json'),'--resume',saved['checkpoint']])
C.write(W/'PRODUCTION_COMMANDS.json',cmds)
launch(cl,cmds,out/('network8_nodes_'+str(int(time.time()))))
