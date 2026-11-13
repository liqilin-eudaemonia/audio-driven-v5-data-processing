"""Audited execution-only repair; never stop nod workers or unrelated GPU jobs."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,locked
from listen_benchmark.j_omni import EXECUTION_REVISION,file_sha
p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','roll','requeue']);p.add_argument('--node');p.add_argument('--gpus',default='0');a=p.parse_args()
if a.action=='freeze':
 assert read(WORK/'validation/processor_cache_regression.json')['passed']
 c=read(WORK/'config.json');rev=WORK/'revisions'/EXECUTION_REVISION;rev.mkdir(parents=True,exist_ok=True)
 if not (rev/'config-before.json').exists():write(rev/'config-before.json',c)
 c['execution_revision']=EXECUTION_REVISION;c['code_hashes']={rel:file_sha(ROOT/rel) for rel in c['code_hashes']}
 write(WORK/'config.json',c);write(rev/'config-after.json',c)
 for rel in c['code_hashes']:
  dst=rev/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
 print('frozen',EXECUTION_REVISION)
elif a.action=='roll':
 cfg=read(WORK/'config.json');node=cfg['nodes'][a.node];gpus=[int(x) for x in a.gpus.split(',')];assert set(gpus)<=set(node['gpu_ids'])
 code=f'ROOT={str(ROOT)!r}\nWORK={str(WORK)!r}\nNODE={a.node!r}\nGPUS={gpus!r}\nREV={EXECUTION_REVISION!r}\n'+r'''
import sys,os,signal,time,socket
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_benchmark.j_annotation import read,write
from listen_j_annotation_supervisor import alive
work=Path(WORK);assert socket.gethostname()==read(work/'config.json')['nodes'][NODE]['hostname']
s=read(work/'nodes'/NODE/'supervisor.json');out=[]
for gpu in GPUS:
 h=read(work/'workers'/f'{NODE}-gpu-{gpu}.json');c=s['children'][f'gpu-{gpu}']
 if h.get('execution_revision')==REV:continue
 if alive(c):
  cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
  assert 'listen_j_omni.py' in cmd and WORK in cmd and 'worker' in cmd
  os.kill(c['pid'],signal.SIGTERM);out.append(dict(gpu=gpu,child=c))
write(work/'revisions'/REV/f'roll-{NODE}-{int(time.time())}.json',dict(at=time.time(),signaled=out))
print('graceful signals',NODE,[x['gpu'] for x in out])
'''
 r=subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+node['host'],'python3 -'],input=code,text=True,capture_output=True,timeout=30);print(r.stdout,r.stderr);raise SystemExit(r.returncode)
else:
 cfg=read(WORK/'config.json')
 for n,c in cfg['nodes'].items():
  for g in c['gpu_ids']:
   hp=WORK/'workers'/f'{n}-gpu-{g}.json'
   if not hp.exists():continue
   h=read(hp);assert h.get('execution_revision')==EXECUTION_REVISION and time.time()-h['heartbeat']<45,(n,g,h['stage'])
 with locked(WORK/'queues/caption/state.lock'):
  s=read(WORK/'queues/caption/state.json');failed=s['failed'];write(WORK/'revisions'/EXECUTION_REVISION/f'failed-before-requeue-{int(time.time())}.json',failed)
  for jid,e in list(failed.items()):
   assert 'Failed to apply Qwen3OmniMoeProcessor' in e['error'],(jid,e['error'][:200])
   # An unlocked inflight claim is durable and recoverable. No journal edits.
   s['inflight'][jid]=dict(worker='execution-fix-requeue',pid=0,at=time.time());s['attempts'].pop(jid,None)
  s['failed']={};write(WORK/'queues/caption/state.json',s)
 print('requeued',list(failed))
