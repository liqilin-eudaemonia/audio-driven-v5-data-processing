from pathlib import Path
import os,sys,json,time,hashlib,fcntl,subprocess
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from vico_more_common import sha,read
from functools import lru_cache
@lru_cache(maxsize=1)
def protocol_sha():return sha(WORK/"protocol.json")

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def folder(method,s):return WORK/method/s['dataset']/s['id']
def valid(method,s,code=None):
    path=folder(method,s)/'complete.json'
    if not path.exists():return False
    r=read(path)
    if r['protocol_sha256']!=protocol_sha():raise ValueError('Protocol mismatch '+str(path))
    if code and r.get('code_sha256')!=code:return False
    return all(Path(p).is_file() and sha(p)==h for p,h in r['files'].items())

@lru_cache(maxsize=1)
def selected(scope='quick200'):
    active=read(WORK/'RUN_SCOPE.json');assert scope==active['scope']=='quick200'
    allowed=set(active['keys'])
    result=[s for s in read(WORK/'protocol.json')['samples'] if s['key'] in allowed]
    assert len(result)==400 and all(s['quick200'] for s in result)
    return result

GPU_UUIDS={4:'GPU-c0ca8be5-8935-4e6c-d3d7-a2c5b58c37c0',7:'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'}

def device():
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in {'4','7'}, 'This task is authorized only on node11 physical GPU4/GPU7'
    row=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==GPU_UUIDS[int(physical)],row
    assert int(row[1])>=32768,row
    d=dict(host=os.uname().nodename,physical_gpu=int(physical),uuid=row[0].strip(),free_mib=int(row[1]),pid=os.getpid(),required_free_gib=32,policy='User-authorized node11 GPU4/GPU7; preserve other jobs; capacity admission')
    import torch
    torch.set_num_threads(2);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.10,0)
    save(WORK/'devices'/f'{os.getpid()}.json',d)
    return d

def pixels_video(path,frames,fps,size=512):
    from vico_avatarforcing import export_rgb
    export_rgb(path,frames,size,fps,18)

def publish(method,s,files,**kw):
    r=dict(id=s['id'],key=s['key'],model=method,protocol_sha256=protocol_sha(),files={str(p):sha(p) for p in files},input_image_sha256=s['reference_sha256'],input_audio_sha256=s['audio_sha256'],source_only=True,listener_audio_generated=False,at=time.time(),**kw)
    save(folder(method,s)/'complete.json',r);return r

def locked_jobs(method,scope='quick200',limit=None,dataset=None):
    count=0
    for s in selected(scope):
        if dataset is not None and s['dataset']!=dataset:continue
        if valid(method,s):continue
        f=folder(method,s);f.mkdir(parents=True,exist_ok=True)
        with (f/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if valid(method,s):continue
            yield s,f
            count+=1
            if limit and count>=limit:return
