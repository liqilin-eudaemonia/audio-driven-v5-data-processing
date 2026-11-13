import concurrent.futures,hashlib,urllib.request,time,json
from pathlib import Path
p=Path(__file__).resolve().parent/'weights';p.mkdir(exist_ok=True)
total=2589355927;expected='b9af0f6c1d732357af52576fde7c2be5cf4bb4ff9df6a8b4370dd4115a32acd6'
u='https://huggingface.co/ZiqiaoPeng/DualTalk/resolve/main/DualTalk.pth?download=true'
partial=p/'DualTalk.partial';start=partial.stat().st_size if partial.exists() else 0
step=64*1024*1024

def run(start):
 end=min(start+step,total)-1;dest=p/f'part_{start}_{end}'
 if dest.exists() and dest.stat().st_size==end-start+1:return dest
 for attempt in range(3):
  try:
   req=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=45) as r:
    assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{total}'
    data=r.read();assert len(data)==end-start+1;dest.write_bytes(data)
   print('downloaded',start,end,flush=True);return dest
  except Exception:
   if attempt==2:raise
 return dest
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(run,range(start,total,step)))
h=hashlib.sha256();dest=p/'DualTalk.pth'
with dest.with_suffix('.assembling').open('wb') as f:
 for path in ([partial] if start else [])+parts:
  with path.open('rb') as s:
   for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b);f.write(b)
assert h.hexdigest()==expected,h.hexdigest();dest.with_suffix('.assembling').rename(dest)
(p/'DOWNLOAD.json').write_text(json.dumps({'source':u,'bytes':total,'sha256':h.hexdigest(),'verified_hf_lfs':True},indent=2))
for path in ([partial] if start else [])+parts:path.unlink()
print('Verified entire checkpoint',expected,flush=True)
