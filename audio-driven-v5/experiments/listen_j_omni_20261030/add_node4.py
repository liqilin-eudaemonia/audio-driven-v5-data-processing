"""User-authorized expansion to node4 GPUs 0..7; no new source partition."""
import json,subprocess,sys,socket,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni import file_sha
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
def remote(host,code):
 r=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+host,'python3 -'],input=code,text=True,capture_output=True,timeout=40)
 print(r.stdout,r.stderr);r.check_returncode()
cfg=read(WORK/'config.json');assert 'node4' not in cfg['nodes']
# Producer code must know compute-only nodes before changing the shared config.
for name,n in cfg['nodes'].items():
 code=f'ROOT={str(ROOT)!r}\nWORK={str(WORK)!r}\nNODE={name!r}\n'+r'''
import sys,os,signal,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read
s=read(Path(WORK)/'nodes'/NODE/'supervisor.json');c=s['children'].get('producer')
if alive(c):
 cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert 'listen_j_omni.py producer' in cmd and WORK in cmd
 os.kill(c['pid'],signal.SIGTERM)
 for _ in range(12):
  if not alive(c):break
  time.sleep(1)
 else:raise RuntimeError('Producer did not stop')
print('producer updated on',NODE)
'''
 remote(n['host'],code)
write(WORK/'revisions/expansion-node4-before.json',cfg)
cfg['nodes']['node4']=dict(host='172.29.225.51',hostname='dedicated-developjob-lq-dev-machine-3-zhw7q',resource_work=str(WORK/'resources_node4'),gpu_ids=list(range(8)),memory=.85,compute_only=True)
assert all(file_sha(ROOT/p)==h for p,h in cfg['code_hashes'].items())
write(WORK/'config.json',cfg)
remote('172.29.225.51',f'ROOT={str(ROOT)!r}\nWORK={str(WORK)!r}\n'+r'''
import sys,os,socket,subprocess,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import spawn,alive
from listen_benchmark.j_annotation import write,read
work=Path(WORK);assert socket.gethostname()==read(work/'config.json')['nodes']['node4']['hostname']
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(float(x.split(',')[1])<2000 for x in raw.strip().splitlines()),raw
w=work/'nodes/node4';existing=read(w/'watchdog.json') if (w/'watchdog.json').exists() else None
assert not alive(existing)
env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
c=spawn(w,'watchdog-bootstrap',['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/listen_j_omni.py','watchdog','--work',WORK,'--node','node4'],env)
write(work/'revisions/expansion-node4.json',dict(at=time.time(),node='node4',gpus=list(range(8)),initial_gpu_memory=raw,watchdog=c))
print('node4 eight-card supervisor launched',c['pid'])
''')
