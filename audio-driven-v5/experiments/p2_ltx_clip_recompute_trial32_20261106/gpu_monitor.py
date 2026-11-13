from pathlib import Path
import runpy,sys,time,json,concurrent.futures
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
W=Path(__file__).resolve().parent;O=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'
m=runpy.run_path(str(O/'monitor.py'),run_name='read_only_probe');cluster=json.loads((O/'cluster.json').read_text())
from p2_listen.contracts import atomic_json
while not any((W/n).exists() for n in ['RESULT.json','ERROR.json','CONTROLLER_ERROR.json']):
 with concurrent.futures.ThreadPoolExecutor(4) as pool:nodes=list(pool.map(lambda n:m['sample_node'](cluster,n),cluster['nodes']))
 value=dict(at=time.time(),nodes=nodes);atomic_json(W/'GPU_STATUS.json',value)
 with (W/'gpu_history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 time.sleep(15)
