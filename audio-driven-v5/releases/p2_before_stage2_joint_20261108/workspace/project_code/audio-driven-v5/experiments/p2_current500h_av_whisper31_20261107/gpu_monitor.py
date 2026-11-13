from pathlib import Path
import runpy,json,time
P=Path(__file__).resolve().parent
m=runpy.run_path(str(P/'coordinator.py'),run_name='resource_functions_only')
while not (P/'GPU_MONITOR_STOP').exists():
 try:
  cluster=m['read'](P/'cluster.json');nodes=m['selected_resources'](cluster)
  metrics=m['read'](P/'PERFORMANCE_STATUS.json')if (P/'PERFORMANCE_STATUS.json').exists()else{}
  v=dict(at=time.time(),step=metrics.get('step'),nodes=nodes,reserved_gpu='node11:7')
  m['atomic_json'](P/'GPU_STATUS.json',v)
  with (P/'gpu_status.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
 except Exception as e:m['atomic_json'](P/'GPU_MONITOR_ERROR.json',dict(at=time.time(),error=repr(e)))
 time.sleep(30)
