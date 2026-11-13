"""Read official tar index by HTTP ranges, download only Conan test arrays."""
import io,requests,tarfile,json,shutil
from pathlib import Path
class HTTPFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.s=requests.Session();r=self.s.head(url,timeout=30);r.raise_for_status();self.size=int(r.headers['Content-Length']);self.pos=0
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def tell(self):return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  r=self.s.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=90);r.raise_for_status()
  if r.status_code!=206:raise RuntimeError('Server ignored Range')
  self.pos+=len(r.content);return r.content
base=Path(__file__).resolve().parents[1]/'learning2listen/src'
f=HTTPFile('http://learning2listen.berkeleyvision.org/conan_data.tar')
with tarfile.open(fileobj=f,mode='r:') as t:
 members=t.getmembers();print('tar members',[(m.name,m.size) for m in members],flush=True)
 for m in members:
  if m.isfile() and '/test/' in m.name and Path(m.name).name in {'p0_list_faces_clean_deca.npy','p1_speak_audio_clean_deca.npy','p1_speak_faces_clean_deca.npy','p1_speak_files_clean_deca.npy'}:
   target=base/'data'/m.name.removeprefix('./').removeprefix('data/')
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.stat().st_size==m.size:continue
   print('Downloading',m.name,m.size,flush=True)
   # bounded streaming Range instead of caching a giant member
   with requests.get(f.url,headers={'Range':f'bytes={m.offset_data}-{m.offset_data+m.size-1}'},stream=True,timeout=90) as r:
    r.raise_for_status();assert r.status_code==206
    with target.open('wb') as out:
     for chunk in r.iter_content(8*1024*1024):out.write(chunk)
print('COMPLETE',flush=True)
