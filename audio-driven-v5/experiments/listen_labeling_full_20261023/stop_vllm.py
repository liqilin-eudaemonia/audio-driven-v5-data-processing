import json,os,signal,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 r=json.loads((root/(name+'.vllm.process.json')).read_text());pid=r['pid'];p=Path('/proc')/str(pid)/'cmdline'
 if p.exists() and p.read_bytes():
  if r['served'].encode() not in p.read_bytes() or os.getpgid(pid)!=pid:raise RuntimeError('Process identity/group mismatch')
  os.killpg(pid,signal.SIGTERM)
  for _ in range(50):
   if not p.exists() or not p.read_bytes():break
   time.sleep(.1)
  if p.exists() and p.read_bytes() and r['served'].encode() in p.read_bytes():
   os.killpg(pid,signal.SIGKILL)
   time.sleep(1)
 print(name,'stopped',pid)
