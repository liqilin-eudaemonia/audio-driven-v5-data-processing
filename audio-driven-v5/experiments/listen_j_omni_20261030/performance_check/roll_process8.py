"""Promote validated 8-process preparation only on CPU-spare node4."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,locked
p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--gpus',required=True);a=p.parse_args();assert a.node=='node4','8-process rollout limited to measured CPU-spare node4';gpus=[int(x) for x in a.gpus.split(',')]
r=read(W/'performance_check/e2e_process8/complete.json');assert r['safe_speed_gain'] and r['normal_outputs']==r['total_outputs'] and r['relative_gain']>=.1
with locked(W/'config-edit.lock'):
 cfg=read(W/'config.json');assert a.node in cfg['nodes'] and set(gpus)<=set(cfg['nodes'][a.node]['gpu_ids'])
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cfg['code_hashes'].items())
 rev=W/'revisions/process8_rollout';rev.mkdir(exist_ok=True)
 if not (rev/'config-before.json').exists():write(rev/'config-before.json',cfg)
 for g in gpus:cfg.setdefault('preparation_profiles',{})[f'{a.node}-gpu-{g}']=r['profiles'][1]
 write(W/'config.json',cfg);write(rev/'config-latest.json',cfg)
code=f'ROOT={str(ROOT)!r}\nWORK={str(W)!r}\nNODE={a.node!r}\nGPUS={gpus!r}\n'+r'''
import os,sys,signal,socket,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read,write
w=Path(WORK);cfg=read(w/'config.json');assert socket.gethostname()==cfg['nodes'][NODE]['hostname'];s=read(w/'nodes'/NODE/'supervisor.json');actions=[]
for g in GPUS:
 c=s['children']['gpu-'+str(g)];h=read(w/'workers'/f'{NODE}-gpu-{g}.json')
 if h.get('preparation_profile')==cfg['preparation_profiles'][f'{NODE}-gpu-{g}'] and h.get('pid')==c['pid'] and alive(c):continue
 assert alive(c),(NODE,g,'worker not live; inspect instead of blind restart')
 cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert 'listen_j_omni.py worker' in cmd and WORK in cmd
 os.kill(c['pid'],signal.SIGTERM);actions.append(dict(gpu=g,child=c,previous_worker=h))
write(w/'revisions/process8_rollout'/f'{NODE}-{time.time_ns()}.json',dict(at=time.time(),actions=actions));print(NODE,'signaled',[x['gpu'] for x in actions])
'''
host=cfg['nodes'][a.node]['host'];cmd=['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+host,'python3 -']
x=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=30);print(x.stdout,x.stderr);x.check_returncode()
