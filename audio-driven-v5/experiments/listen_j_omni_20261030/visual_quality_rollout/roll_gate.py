"""Restart only explicitly selected owned workers after actual-media validation."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

W=Path(__file__).resolve().parents[1]
ROOT=W.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read

p=argparse.ArgumentParser()
p.add_argument('--node',required=True)
p.add_argument('--gpus',required=True)
a=p.parse_args()
gpus=[int(g) for g in a.gpus.split(',')]
cfg=read(W/'config.json')
assert a.node in cfg['nodes'] and set(gpus)<=set(cfg['nodes'][a.node]['gpu_ids'])
assert read(W/'visual_quality_rollout/node4_parity.json')['passed']
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cfg['code_hashes'].items())
code=f'ROOT={str(ROOT)!r}\nWORK={str(W)!r}\nNODE={a.node!r}\nGPUS={gpus!r}\n'+r'''
import os,sys,socket,signal,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read,write
w=Path(WORK);cfg=read(w/'config.json')
assert socket.gethostname()==cfg['nodes'][NODE]['hostname']
s=read(w/'nodes'/NODE/'supervisor.json');actions=[]
for gpu in GPUS:
 child=s['children']['gpu-'+str(gpu)]
 h=read(w/'workers'/f'{NODE}-gpu-{gpu}.json')
 assert alive(child) and h['pid']==child['pid'], 'Inspect missing/mismatched worker instead of blind restart'
 if 'visual-gate-v1' in h.get('execution_revision',''):continue
 command=Path(f"/proc/{child['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert 'listen_j_omni.py worker' in command and WORK in command
 os.kill(child['pid'],signal.SIGTERM)
 actions.append(dict(gpu=gpu,child=child,previous_worker=h))
write(w/'visual_quality_rollout/restarts'/f'{NODE}-{time.time_ns()}.json',dict(at=time.time(),actions=actions))
print(NODE,'signaled',[x['gpu'] for x in actions])
'''
cmd=['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519',
     '-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600',
     '-o','ControlPath=/tmp/jomni-guard-%C','root@'+cfg['nodes'][a.node]['host'],'python3 -']
r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=30)
print(r.stdout,r.stderr)
r.check_returncode()
