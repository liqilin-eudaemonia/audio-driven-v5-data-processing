from pathlib import Path
import sys,copy,json
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from listen_cluster import launch,distributed_commands
base=json.loads((ROOT/'experiments/p2_ltx_all_lora_noevent24_after6000_20261109/cluster.json').read_text())
for name,hca in [('single','=mlx5_0:1'),('eight','=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1')]:
 c=copy.deepcopy(base);c['master_port']=29927 if name=='single'else 29928;c['network'].update(NCCL_IB_HCA=hca,PROBE_RESULT=str(W/(name+'_NETWORK.json')))
 try:launch(c,distributed_commands(c,W/'network_probe.py',[]),W/(name+'_network_logs'))
 except Exception as e:(W/(name+'_NETWORK_ERROR.json')).write_text(json.dumps(dict(error=repr(e))))
