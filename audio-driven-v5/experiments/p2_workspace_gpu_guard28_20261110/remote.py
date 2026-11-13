"""Node-side read-only ownership scan and twice-checked GPU job termination."""
from pathlib import Path
import os,json,time,signal,subprocess
from policy import classify,task_paths

def stat(pid):
 s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=s[19],ppid=int(s[1]),state=s[0])
def alive(r):
 try:s=stat(r['pid']);return s['start_ticks']==r['start_ticks']and s['state']!='Z'
 except (FileNotFoundError,ProcessLookupError):return False

def ancestry(pid):
 result=[];complete=False;seen=set()
 for _ in range(16):
  if pid in (0,1):complete=True;break
  if pid in seen:break
  seen.add(pid)
  try:
   s=stat(pid);d=Path('/proc',str(pid));cwd=os.readlink(d/'cwd');argv=[v.decode(errors='replace')for v in(d/'cmdline').read_bytes().split(b'\0')if v]
   result.append(dict(**s,cwd=cwd,comm=(d/'comm').read_text().strip(),task_paths=task_paths(argv,cwd)));pid=s['ppid']
  except (FileNotFoundError,PermissionError,ProcessLookupError,OSError)as e:
   result.append(dict(pid=pid,error=type(e).__name__));break
 return result,complete

def gpu_state():
 cards={}
 for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10).splitlines():
  a=[p.strip()for p in row.split(',')];cards[a[1]]=int(a[0])
 apps={}
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True,timeout=10).splitlines():
  a=[p.strip()for p in row.split(',')]
  if len(a)==2 and a[0].isdigit()and a[1]in cards:apps.setdefault(int(a[0]),set()).add(cards[a[1]])
 return apps

def scan(x):
 selected=set(x['gpus']);protected={(p['pid'],p['start_ticks'])for p in x['protected']};out=[]
 for pid,gpus in gpu_state().items():
  if not selected&gpus:continue
  try:r=stat(pid)
  except (FileNotFoundError,ProcessLookupError):continue
  is_protected=(pid,r['start_ticks'])in protected
  if is_protected:chain=[];complete=True
  else:chain,complete=ancestry(pid)
  owner,reason=classify(chain,x['workspace_root'],complete,is_protected)
  # Do not signal a process spanning the reserved GPU boundary.
  if owner=='foreign'and not gpus<=selected:owner='unknown';reason='also_owns_reserved_gpu'
  out.append(dict(**r,gpus=sorted(gpus),owner=owner,reason=reason,chain=chain))
 return dict(at=time.time(),records=out)

def act(x):
 fresh=scan(x);by_pid={r['pid']:r for r in fresh['records']};results=[];targets=[]
 for old in x['expected']:
  r=by_pid.get(old['pid'])
  if not r or r['start_ticks']!=old['start_ticks']or r['gpus']!=old['gpus']or r['owner']!='foreign':
   results.append(dict(pid=old['pid'],skipped='changed_or_no_longer_foreign'));continue
  if not alive(r):continue
  # Current node-side attribution and GPU ownership match the earlier read pass.
  os.kill(r['pid'],signal.SIGTERM);targets.append(r);results.append(dict(pid=r['pid'],sigterm=True))
 until=time.time()+5
 while any(alive(r)for r in targets)and time.time()<until:time.sleep(.2)
 remaining=[r for r in targets if alive(r)]
 verified={r['pid']:r for r in scan(x)['records']}if remaining else{}
 for r in remaining:
  v=verified.get(r['pid'])
  if v and v['start_ticks']==r['start_ticks']and v['gpus']==r['gpus']and v['owner']=='foreign'and alive(v):
   os.kill(r['pid'],signal.SIGKILL);next(z for z in results if z['pid']==r['pid'])['sigkill']=True
  else:next(z for z in results if z['pid']==r['pid'])['force_skipped']='ownership_changed'
 return dict(at=time.time(),results=results)
