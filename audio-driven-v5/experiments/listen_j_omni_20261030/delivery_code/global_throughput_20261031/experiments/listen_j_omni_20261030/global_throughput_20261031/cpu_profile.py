import sys,json,concurrent.futures,time
from pathlib import Path
w=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_omni_20261030');sys.path.insert(0,str(w/'nod_cpu_share'));from probe import remote
code='''import time,os,json,collections
from pathlib import Path
def snap():
 r={}
 for pid in pids:
  p=Path('/proc')/str(pid)
  try:
   s=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();r[int(p.name)]=(int(s[1]),(int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'),cmd)
  except (OSError,ValueError):pass
 return r
w=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_omni_20261030');pids=set();assigned={}
def add(pid,g):
 if pid in pids:return
 pids.add(pid);assigned[pid]=g
 try:
  for task in (Path('/proc')/str(pid)/'task').iterdir():
   for child in (task/'children').read_text().split():add(int(child),g)
 except OSError:pass
for gpu in range(8):add(json.loads((w/'workers'/f'{NODE}-gpu-{gpu}.json').read_text())['pid'],'omni')
add(json.loads((w/'visual_quality/nodes'/NODE/'controller.json').read_text())['pid'],'quality')
a=snap();t=time.monotonic();time.sleep(10);b=snap();dt=time.monotonic()-t;groups=collections.defaultdict(float);counts=collections.Counter();tops=[]
def group(pid,depth=0):
 if pid not in b or depth>6:return assigned.get(pid,'other')
 pp,c,cmd=b[pid]
 for token,label in [('listen_j_corrected_nod.py','corrected_nod'),('listen_j_visual_gate_cluster.py','quality'),('listen_j_visual_quality_early.py','early_quality'),('listen_j_annotation_nod.py','upstream_nod'),('listen_j_omni.py worker','omni')]:
  if token in cmd:return label
 return group(pp,depth+1)
for pid,(pp,c,cmd) in b.items():
 if pid in a:
  cores=(c-a[pid][1])/dt;g=group(pid);groups[g]+=cores;counts[g]+=1
  if cores>.3:tops.append((round(cores,2),g,pid,cmd[:160]))
print(json.dumps(dict(groups=groups,counts=counts,top=sorted(tops,reverse=True)[:12])))'''
with concurrent.futures.ThreadPoolExecutor(4) as p:r=dict(zip(['node17','node8','node4','node12'],p.map(lambda n:remote(n,"NODE="+repr(n)+"\n"+code),['node17','node8','node4','node12'])))
(w/'global_throughput_20261031/cpu_groups.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
