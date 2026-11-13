"""Independent, exact6000 native save/stop request. Never signal GPU jobs."""
from pathlib import Path
import fcntl,json,os,time,hashlib
WORK=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def record():
 s=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=os.getpid(),start_ticks=s[19],at=time.time())
def request_stop(path,owner,**extra):
 try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 except FileExistsError:return False
 with os.fdopen(fd,'w')as f:json.dump(dict(owner=owner,at=time.time(),**extra),f);f.flush();os.fsync(f.fileno())
 return True
def main():
 p=read(WORK/'PROTOCOL.json');out=Path(p['full_parent'])
 with(WORK/'gate.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(WORK/'GATE_CONTROLLER.json',record());last=0
  while not(WORK/'STOP').exists():
   step=read(out/'status.json')['step']
   if step>=p['target_step']:
    assert hashlib.sha256((out/'config.json').read_bytes()).hexdigest()==p['full_config_sha256']
    created=request_stop(out/'STOP',p['owner'],target_step=6000,reason='Authorized resource swap after complete6000 save')
    write(WORK/'GATE_STOP_REQUESTED.json',dict(at=time.time(),observed_step=step,created=created,stop=read(out/'STOP')));return
   if time.time()-last>5:write(WORK/'GATE_STATUS.json',dict(at=time.time(),stage='waiting_exact6000',step=step,target=6000));last=time.time()
   time.sleep(.1)
if __name__=='__main__':main()
