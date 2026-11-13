import json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');w=ROOT/'experiments/listen_j_omni_20261030';q=w/'visual_quality'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from listen_benchmark.j_annotation import read,write,locked
from listen_j_annotation_supervisor import alive,spawn
from listen_j_omni import environment,PYTHON
host='dedicated-developjob-lq-dev-machine-1-4btg8'
assert socket.gethostname()==host
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
dev={int(a):int(b) for a,b in (line.split(',') for line in raw.strip().splitlines())}
old=read(w/'nodes/node12/watchdog.json') if (w/'nodes/node12/watchdog.json').exists() else None
if not alive(old):assert len(dev)==8 and all(v<2000 for v in dev.values()),dev
with locked(w/'config-edit.lock'):
 cfg=read(w/'config.json')
 cfg['nodes']['node12']=dict(host='172.29.252.37',hostname=host,resource_work=str(w/'resources_node12'),gpu_ids=list(range(8)),memory=.85,compute_only=True,ssh_key='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1-4btg8_ed25519')
 for g in range(8):cfg['preparation_profiles'][f'node12-gpu-{g}']=dict(backend='process',workers=4,prefetch=8,ready_first=True)
 write(w/'config.json',cfg)
local=q/'nodes/node12';local.mkdir(parents=True,exist_ok=True)
quality=read(local/'watchdog.json') if (local/'watchdog.json').exists() else None
if not alive(quality):
 quality=spawn(local,'watchdog-launch',[str(w/'visual_quality_rollout/runtime_env/bin/python'),str(ROOT/'scripts/listen_j_visual_gate_cluster.py'),'watchdog','--work',str(q),'--node','node12','--workers','48'],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'))
if not alive(old):old=spawn(w/'nodes/node12','watchdog-launch',[PYTHON,str(ROOT/'scripts/listen_j_omni.py'),'watchdog','--work',str(w),'--node','node12'],environment())
result=dict(at=time.time(),hostname=host,quality=quality,omni_watchdog=old,devices_before=dev)
write(w/'visual_quality_rollout/node12_bootstrap.json',result);print(json.dumps(result))
