"""Promote only exact-input, end-to-end validated queued submission profiles."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,locked
p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--gpus',required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
gpus=[int(x) for x in a.gpus.split(',')];e=read(a.evidence)
assert e['safe_speed_gain'] and e.get('adaptive') and e['speedup']>=1.1
assert all(all(r[k] for k in ['all_four_captions_identical','all_prompt_tokens_identical','all_normal','all_match_reference']) for r in e['comparisons'])
with locked(W/'config-edit.lock'):
 cfg=read(W/'config.json');assert set(gpus)<=set(cfg['nodes'][a.node]['gpu_ids']);assert e['pipeline_id']==cfg['pipeline_id']
 for g in gpus:
  name=f'{a.node}-gpu-{g}';assert cfg['preparation_profiles'][name]==e['profile'],'Profile needs its own end-to-end validation'
  assert cfg.get('engine_max_num_seqs',{}).get(name,1)==1
  cfg.setdefault('queued_requests_per_call',{})[name]=4;cfg.setdefault('adaptive_request_groups',{})[name]=True
 for rel in cfg['code_hashes']:assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==cfg['code_hashes'][rel],rel
 write(W/'config.json',cfg)
code=f'ROOT={str(ROOT)!r}\nWORK={str(W)!r}\nNODE={a.node!r}\nGPUS={gpus!r}\nEVIDENCE={str(a.evidence.resolve())!r}\n'+r'''
import os,sys,socket,signal,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read,write
w=Path(WORK);cfg=read(w/'config.json');assert socket.gethostname()==cfg['nodes'][NODE]['hostname']
s=read(w/'nodes'/NODE/'supervisor.json');actions=[]
for gpu in GPUS:
 child=s['children']['gpu-'+str(gpu)];h=read(w/'workers'/f'{NODE}-gpu-{gpu}.json')
 assert alive(child) and h['pid']==child['pid']
 if h.get('production_batch_size')==4 and h.get('adaptive_request_groups'):continue
 command=Path(f"/proc/{child['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
 assert 'listen_j_omni.py worker' in command and WORK in command
 os.kill(child['pid'],signal.SIGTERM);actions.append(dict(gpu=gpu,child=child,previous=h))
write(w/'visual_quality_rollout/queue_restarts'/f'{NODE}-{time.time_ns()}.json',dict(at=time.time(),actions=actions,evidence=EVIDENCE))
print(NODE,'queued-submission promotion signaled',[x['gpu'] for x in actions])
'''
import socket
nc=cfg['nodes'][a.node]
if socket.gethostname()==nc['hostname']:cmd=[sys.executable,'-']
else:cmd=['ssh','-i',nc.get('ssh_key','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+nc['host'],'python3 -']
r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=30);print(r.stdout,r.stderr);r.check_returncode()
