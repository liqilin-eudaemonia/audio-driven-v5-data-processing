"""Bounded live CPU allocation trial; always restores the prior allocation."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
A=ROOT/'experiments/listen_j_annotation_node17_20261029'
Q=ROOT/'data_corpora/seamless_listen_j_visual_quality_v1_20261030/node17'
HOSTS={'node17':('172.29.15.91','/tmp/listen-node17-speed-ssh.sock'),'node8':('172.29.104.233','/tmp/listen-node8-annotation-ssh.sock')}

def read(p):return json.loads(p.read_text())
def write(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def remote(node,code):
 host,sock=HOSTS[node]
 p=subprocess.run(['ssh','-S',sock,'-F','/dev/null','-o','ConnectTimeout=10','root@'+host,'python3 -'],input=code,text=True,capture_output=True,timeout=35,check=True)
 return json.loads(p.stdout)

def history(path,since):
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2000000));raw=f.read()
 rows=[]
 for line in raw.splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('at',r.get('heartbeat',0))>=since:rows.append(r)
 return rows

INSPECT=r'''
import json,time,subprocess
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
w=root/'experiments'/('listen_j_annotation_node17_20261029' if NODE=='node17' else 'listen_j_annotation_node8_20261030')
q=root/'data_corpora/seamless_listen_j_visual_quality_v1_20261030'/NODE
read=lambda p:json.loads(p.read_text())
def live(c):
 try:
  s=Path('/proc/%s/stat'%c['pid']).read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and s[19]==str(c['start_ticks'])
 except (OSError,KeyError):return False
s=read(w/'supervisor.json');qc=read(q/'controller.json')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
workers={str(i):read(w/'workers'/f'gpu-{i}.json') for i in range(8)}
print(json.dumps({'at':time.time(),'gpu_util':{int(a):int(b) for a,b in (l.split(',') for l in raw.splitlines())},
 'vlm':read(w/'queues/vlm/state.json')['completed_clips'],'nod':read(w/'queues/nod/state.json')['completed_clips'],
 'quality':read(q/'queues/quality/state.json')['completed_clips'],
 'failures':{b:len(read(w/'queues'/b/'state.json')['failed']) for b in ('vlm','nod')},
 'quality_failures':len(read(q/'queues/quality/state.json')['failed']),
 'nod_live':sum(live(c) for n,c in s['children'].items() if n.startswith('nod-')),
 'quality_live':sum(live(c) for c in qc['children'].values()),
 'gpu_live':sum(live(c) for n,c in s['children'].items() if n.startswith('gpu-')),
 'gpu_pids':{n:c['pid'] for n,c in s['children'].items() if n.startswith('gpu-')},
 'completed_worker_clips':sum(h['completed_clips'] for h in workers.values()),
 'waiting_input':sum(h.get('stage','').startswith('preparing') for h in workers.values()),
 'cpu_stat':Path('/sys/fs/cgroup/cpu.stat').read_text()}))
'''

def inspect(node):return node,remote(node,'NODE='+repr(node)+'\n'+INSPECT)

def collect():
 with concurrent.futures.ThreadPoolExecutor(2) as pool:return dict(pool.map(inspect,HOSTS))

def restore():
 c=read(A/'config.json');c['nod_workers']=read(BASE/'config_before.json')['nod_workers'];write(A/'config.json',c)
 write(Q/'worker_control.json',read(BASE/'quality_control_before.json'))
 write(BASE/'restored.json',{'at':time.time(),'nod_workers':c['nod_workers'],'quality_workers':read(Q/'worker_control.json')['workers']})


def main():
 assert not (BASE/'started.json').exists(),'Run directory already used'
 write(BASE/'config_before.json',read(A/'config.json'));write(BASE/'quality_control_before.json',read(Q/'worker_control.json'))
 write(BASE/'resource_tuning_before.json',read(A/'resource_tuning.json'))
 start=time.time();baseline={}
 for node in HOSTS:
  w=ROOT/'experiments'/('listen_j_annotation_node17_20261029' if node=='node17' else 'listen_j_annotation_node8_20261030')
  baseline[node]={'annotation':history(w/'telemetry.jsonl',start-650),
   'quality':history(ROOT/'data_corpora/seamless_listen_j_visual_quality_v1_20261030'/node/'telemetry.jsonl',start-650)}
 write(BASE/'baseline.json',baseline);write(BASE/'started.json',{'at':start,'node17_nod':64,'node17_quality':16,'duration_after_stable_s':600,'control':'node8'})
 samples=[];stable=None
 try:
  c=read(A/'config.json');c['nod_workers']=64;write(A/'config.json',c)
  control=read(Q/'worker_control.json');control.update(workers=16,reason='Bounded node17 CPU allocation trial; automatic restoration after measurement');write(Q/'worker_control.json',control)
  stopped=remote('node17',r'''
import os,signal,json
from pathlib import Path
w=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_annotation_node17_20261029')
s=json.loads((w/'supervisor.json').read_text());stopped=[]
for i in range(64,80):
 c=s['children'].get('nod-'+str(i),{})
 try:
  f=Path('/proc/%s/stat'%c['pid']).read_text().rsplit(')',1)[1].split()
  cmd=Path('/proc/%s/cmdline'%c['pid']).read_bytes()
  if f[0]!='Z' and f[19]==str(c.get('start_ticks')) and b'listen_j_annotation_nod.py' in cmd:
   os.kill(c['pid'],signal.SIGTERM);stopped.append(c['pid'])
 except (OSError,KeyError):pass
print(json.dumps(stopped))
''')
  write(BASE/'nod_drain_signals.json',stopped)
  while True:
   obs=collect();samples.append(obs);write(BASE/'samples.json',samples)
   trial=obs['node17'];now=time.time()
   if trial['gpu_live']!=8:raise RuntimeError('A GPU worker exited during trial')
   if sum(trial['failures'].values()) or trial['quality_failures']:raise RuntimeError('Queue failure during trial')
   if stable is None and trial['nod_live']==64 and trial['quality_live']==16:
    stable=now;write(BASE/'stable.json',{'at':stable,'first_sample':len(samples)-1})
   write(BASE/'status.json',{'at':now,'phase':'measuring' if stable else 'draining',
     'elapsed_s':now-start,'measurement_s':now-stable if stable else 0,'nodes':obs})
   if stable and now-stable>=600:break
   if not stable and now-start>240:raise RuntimeError('CPU workers did not finish draining')
   if now-start>900:raise RuntimeError('Trial time limit')
   time.sleep(15)
  write(BASE/'finished.json',{'at':time.time(),'samples':len(samples),'measurement_s':time.time()-stable})
 except Exception as e:
  write(BASE/'error.json',{'at':time.time(),'error':repr(e)});raise
 finally:restore()
 print('Trial finished; original CPU allocations restored for review.')

if __name__=='__main__':main()
