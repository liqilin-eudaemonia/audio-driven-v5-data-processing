from pathlib import Path
import os,json,subprocess,sys,time
w=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_upright_av500h_v3_20261105')
root=w.parents[1]
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from listen_benchmark.j_annotation import write
while not ((w/'TASK_COMPLETE.json').exists() and (w/'STATISTICS_AUDIT.json').exists()):
 for name in ('CONTROLLER_ERROR.json','INDEPENDENT_AUDIT_ERROR.json'):
  if (w/name).exists():raise RuntimeError(str(w/name))
 time.sleep(10)
write(w/'FINALIZATION_STATUS.json',dict(stage='independent_final_audit',at=time.time()))
p=subprocess.run([sys.executable,str(root/'scripts/audit_j_upright_av500h_v3.py'),'final'],check=True)
import listen_richness_cluster_control as cluster
body=cluster.REMOTE_COMMON+"""
owned={}
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 p=W/'nodes'/NODE/('production-gpu-'+str(gpu)+'.json')
 owned[str(gpu)]=alive(read(p)) if p.exists() else None
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=20)
print(json.dumps(dict(node=NODE,owned_worker_alive=owned,gpus=devices(),compute_apps=apps)))
"""
released=cluster.all_nodes(w,body)
write(w/'GPU_RELEASE_CHECK.json',dict(at=time.time(),nodes=released,all_owned_workers_exited=all(not live for node in released.values() for live in node['owned_worker_alive'].values())))
write(w/'FINALIZATION_STATUS.json',dict(stage='complete',at=time.time(),independent_final_audit=True))
