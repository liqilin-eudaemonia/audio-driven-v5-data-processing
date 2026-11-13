import os,time,json,subprocess
from pathlib import Path

def read(p):
 try:return Path(p).read_text().strip()
 except OSError:return None
allowed=set(os.sched_getaffinity(0))
def stat():
 cpus={int(f[0][3:]):list(map(int,f[1:9])) for l in read('/proc/stat').splitlines() if (f:=l.split()) and f[0].startswith('cpu') and f[0][3:].isdigit()}
 cg={k:int(v) for k,v in (l.split() for l in read('/sys/fs/cgroup/cpu.stat').splitlines())}
 net={f[0].strip():list(map(int,f[1].split())) for l in read('/proc/net/dev').splitlines()[2:] if len(f:=l.split(':'))==2}
 processes={}
 for pid in [98935,107034,107050,107104,107119,107133,107147,107207]:
  s=read(f'/proc/{pid}/stat')
  if s:
   f=s[s.rfind(')')+2:].split();processes[pid]=dict(ticks=int(f[11])+int(f[12]),name=s[s.find('(')+1:s.rfind(')')])
 return dict(t=time.monotonic(),cpus=cpus,cg=cg,net=net,processes=processes)
def gpu():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 return r.stdout.strip()
a=stat();gpus=[]
for _ in range(3):gpus.append(gpu());time.sleep(4)
b=stat();dt=b['t']-a['t'];delta=[sum(b['cpus'][i][j]-a['cpus'][i][j] for i in allowed) for j in range(8)];total=sum(delta)
mem={l.split(':')[0]:int(l.split()[1]) for l in read('/proc/meminfo').splitlines() if len(l.split())>1}
net=[]
for k,v in b['net'].items():
 if k not in a['net'] or k=='lo':continue
 x=a['net'][k];rx=(v[0]-x[0])/dt/1024**2;tx=(v[8]-x[8])/dt/1024**2
 if rx+tx>.1:net.append(dict(interface=k,rx_MiB_s=round(rx,2),tx_MiB_s=round(tx,2)))
procs=[]
for pid,x in b['processes'].items():
 if pid in a['processes']:procs.append(dict(pid=pid,name=x['name'],cpu_cores=round((x['ticks']-a['processes'][pid]['ticks'])/os.sysconf('SC_CLK_TCK')/dt,2)))
print(json.dumps(dict(utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),sample_seconds=round(dt,2),affinity_cores=len(allowed),affinity_cpu_busy_pct=round(100*(total-delta[3]-delta[4])/total,2),iowait_pct=round(100*delta[4]/total,2),cgroup_cpu_cores_used=round((b['cg']['usage_usec']-a['cg']['usage_usec'])/dt/1e6,2),throttled_usec_delta=b['cg']['throttled_usec']-a['cg']['throttled_usec'],memory_available_GiB=round(mem['MemAvailable']/1024**2,1),loadavg=read('/proc/loadavg'),pressure={k:read('/proc/pressure/'+k) for k in ('cpu','io','memory')},gpu_samples=gpus,network_MiB_s=net,selected_process_cpu=procs)))
