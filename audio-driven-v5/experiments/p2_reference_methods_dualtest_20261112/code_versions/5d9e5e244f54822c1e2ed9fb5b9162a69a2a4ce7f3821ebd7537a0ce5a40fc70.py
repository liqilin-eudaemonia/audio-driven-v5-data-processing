from pathlib import Path
import os,sys,json,time,hashlib,fcntl,subprocess
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from vico_more_common import sha,read,gpu_guard

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def folder(method,s):return WORK/method/s['dataset']/s['id']
def valid(method,s,code=None):
    path=folder(method,s)/'complete.json'
    if not path.exists():return False
    r=read(path)
    if r['protocol_sha256']!=sha(WORK/'protocol.json'):raise ValueError('Protocol mismatch '+str(path))
    if code and r.get('code_sha256')!=code:return False
    return all(Path(p).is_file() and sha(p)==h for p,h in r['files'].items())

def selected(scope='full'):
    return [s for s in read(WORK/'protocol.json')['samples'] if scope=='full' or s['quick200']]

def device():
    d=gpu_guard(32)
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.10,0)
    return d

def pixels_video(path,frames,fps,size=512):
    from vico_avatarforcing import export_rgb
    export_rgb(path,frames,size,fps,18)

def publish(method,s,files,**kw):
    r=dict(id=s['id'],key=s['key'],model=method,protocol_sha256=sha(WORK/'protocol.json'),files={str(p):sha(p) for p in files},input_image_sha256=s['reference_sha256'],input_audio_sha256=s['audio_sha256'],source_only=True,listener_audio_generated=False,at=time.time(),**kw)
    save(folder(method,s)/'complete.json',r);return r

def locked_jobs(method,scope='full',limit=None):
    count=0
    for s in selected(scope):
        if valid(method,s):continue
        f=folder(method,s);f.mkdir(parents=True,exist_ok=True)
        with (f/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if valid(method,s):continue
            yield s,f
            count+=1
            if limit and count>=limit:return
