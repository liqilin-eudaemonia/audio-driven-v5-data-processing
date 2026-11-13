import json,os,socket,subprocess,time,sys
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');w=root/'experiments/listen_j_omni_20261030';q=w/'visual_quality'
assert socket.gethostname()==sys.argv[1]
workers=int(sys.argv[2]);python=str(w/'visual_quality_rollout/runtime_env/bin/python');script=str(root/'scripts/listen_j_visual_gate.py')
r=subprocess.run([python,script,'bootstrap','--work',str(q),'--omni',str(w)],text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr)
c=json.loads(r.stdout)
sys.path.insert(0,str(root/'scripts'))
from listen_j_annotation_supervisor import alive
old=json.loads((q/'watchdog.json').read_text()) if (q/'watchdog.json').exists() else None
if alive(old):raise RuntimeError('Quality watchdog already active')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2')
with (q/'watchdog-launch.log').open('a') as f:
 p=subprocess.Popen([python,script,'watchdog','--work',str(q),'--workers',str(workers)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
print(json.dumps(dict(pid=p.pid,hostname=socket.gethostname(),contract=c['contract_sha256'],workers=workers,at=time.time())))
