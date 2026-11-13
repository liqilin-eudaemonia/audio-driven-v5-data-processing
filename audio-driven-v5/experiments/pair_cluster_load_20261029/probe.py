"""Read-only short resource sample; no model loading or filesystem traversal."""
import os,json,time,subprocess,socket
from pathlib import Path

def read(p):
 try:return Path(p).read_text().strip()
 except OSError:return None

def run(args,timeout=12):
 try:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
  return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
 except Exception as e:return dict(error=str(e))

def sample():
 cpu=list(map(int,read('/proc/stat').splitlines()[0].split()[1:9]))
 return dict(cpu=cpu,net=read('/proc/net/dev'),disk=read('/proc/diskstats'),time=time.monotonic())

a=sample()
gpu=run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'])
gpup=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');args=[x for x in args if x]
  if not args:continue
  scripts=[x for x in args if x.endswith('.py') or x in ('torchrun','torch.distributed.run','vllm.entrypoints.openai.api_server')]
  interesting=any(any(t in x.lower() for t in ('train','torchrun','infer','vllm','cache','ffmpeg','listen','glare')) for x in args[:3])
  if not interesting:continue
  fields={}
  for i,x in enumerate(args[:-1]):
   if x in ('--config','--config_path','--config-path','--output-dir','--output_dir','--work-dir','--log-dir','--model','--nnodes','--nproc_per_node','--nproc-per-node'):fields[x]=args[i+1]
  processes.append(dict(pid=int(p.name),program=Path(args[0]).name,scripts=scripts,fields=fields,cwd=os.readlink(p/'cwd'),stdout=os.readlink(p/'fd/1')))
 except (OSError,ValueError):pass
remaining=10-(time.monotonic()-a['time'])
if remaining>0:time.sleep(remaining)
b=sample();delta=[y-x for x,y in zip(a['cpu'],b['cpu'])];total=sum(delta)
mem={line.split(':')[0]:int(line.split()[1]) for line in read('/proc/meminfo').splitlines() if len(line.split())>1}
cgpaths=['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/cpu.stat','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/cpuset/cpuset.cpus','/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes']
mounts=[]
for line in (read('/proc/mounts') or '').splitlines():
 f=line.split()
 if len(f)>2 and (f[1] in ('/','/private','/public','/private/liqi_workspace') or f[1].startswith('/public/0_DATA')):mounts.append(dict(device=f[0],mount=f[1],type=f[2]))
print(json.dumps(dict(hostname=socket.gethostname(),utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),cpu_count=os.cpu_count(),cpu_affinity_count=len(os.sched_getaffinity(0)),loadavg=read('/proc/loadavg'),sample_seconds=round(b['time']-a['time'],2),cpu_busy_pct=round(100*(total-delta[3]-delta[4])/total,2),cpu_iowait_pct=round(100*delta[4]/total,2),memory_kib={k:mem.get(k) for k in ('MemTotal','MemAvailable','SwapTotal','SwapFree','Dirty','Writeback')},cgroup={p:read(p) for p in cgpaths if Path(p).exists()},pressure={k:read('/proc/pressure/'+k) for k in ('cpu','io','memory')},gpu=gpu,gpu_processes=gpup,processes=processes,top_cpu=run(['ps','-eo','pid,ppid,comm,pcpu,pmem,rss,stat','--sort=-pcpu']),mounts=mounts,network_before=a['net'],network_after=b['net'],disk_before=a['disk'],disk_after=b['disk']),ensure_ascii=False))
