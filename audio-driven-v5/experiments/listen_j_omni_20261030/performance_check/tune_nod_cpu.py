"""Change only this pipeline's CPU nod concurrency; preserve all unfinished jobs."""
import argparse,subprocess,json
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
p=argparse.ArgumentParser();p.add_argument('--node',choices=['node17','node8'],required=True);p.add_argument('--workers',type=int,choices=[48,80],required=True);a=p.parse_args()
cfg=json.loads((W/'config.json').read_text());old=cfg['nodes'][a.node]['old_work'];assert old
code=f'ROOT={str(ROOT)!r}\nWORK={old!r}\nOMNI={str(W)!r}\nNODE={a.node!r}\nTARGET={a.workers!r}\n'+r'''
import os,sys,signal,socket,time
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read,write,locked
w=Path(WORK);ow=Path(OMNI);assert socket.gethostname()==read(ow/'config.json')['nodes'][NODE]['hostname']
with locked(ow/'nod-cpu-config.lock'):
 c=read(w/'config.json');before=c.get('nod_workers');assert before in (48,80),before
 state=read(w/'supervisor.json');retire=[]
 if TARGET<before:
  for slot in range(TARGET,before):
   child=state['children'].get(f'nod-{slot}')
   if not child or not alive(child):continue
   argv=Path(f"/proc/{child['pid']}/cmdline").read_bytes().split(b'\x00');argv=[x.decode() for x in argv if x]
   assert str(Path(ROOT)/'scripts/listen_j_annotation_nod.py') in argv and argv[argv.index('--work')+1]==WORK and argv[argv.index('--slot')+1]==str(slot),argv
   retire.append(dict(slot=slot,child=child))
 record=dict(at=time.time(),node=NODE,previous=before,target=TARGET,config_before=c.copy(),retiring=retire)
 c['nod_workers']=TARGET;write(w/'config.json',c)
 for item in retire:
  child=item['child']
  if alive(child):os.kill(child['pid'],signal.SIGTERM)
 write(ow/'revisions/nod_cpu_balance'/f'{NODE}-{time.time_ns()}.json',record)
 print(NODE,'nod CPU cap',before,'->',TARGET,'graceful retirement',len(retire),flush=True)
'''
x=subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+cfg['nodes'][a.node]['host'],'python3 -'],input=code,text=True,capture_output=True,timeout=30);print(x.stdout,x.stderr);x.check_returncode()
