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
    return all(verified_file(p,h) for p,h in r['files'].items())

SCOPE='responsenet586'
TOTAL=586
@lru_cache(maxsize=1)
def selected(scope=SCOPE):
    active=read(WORK/'RUN_SCOPE.json');assert scope==active['scope']==SCOPE
    result=read(WORK/'protocol.json')['samples']
    assert len(result)==len({s['key'] for s in result})==TOTAL
    assert all(s['dataset']=='responsenet' for s in result)
    assert [s['key'] for s in result]==active['keys']
    return result

@lru_cache(maxsize=1)
def groups():
    bykey={s['key']:s for s in selected()}
    return [[bykey[k] for k in batch] for batch in read(WORK/'BATCHES.json')['groups']]

def activate_readers():
    path=str(WORK/'pinned_metrics')
    if path not in sys.path:sys.path.insert(0,path)
    from evaluation.seamless import reader
    assert WORK/'pinned_metrics' in Path(reader.__file__).parents
    reader.DATA_ROOT=ROOT
    return reader

@lru_cache(maxsize=131072)
def _version_sha(path,size,mtime,ctime):return sha(path)
def verified_file(path,expected):
    path=Path(path)
    if not path.is_file():return False
    st=path.stat()
    return _version_sha(str(path),st.st_size,st.st_mtime_ns,st.st_ctime_ns)==expected

GPU_UUIDS={7:'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'}

def device():
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical=='7', 'Latest user allocation: node11 GPU7 only; GPU4 forbidden'
    row=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==GPU_UUIDS[int(physical)],row
    assert int(row[1])>=32768,row
    d=dict(host=os.uname().nodename,physical_gpu=int(physical),uuid=row[0].strip(),free_mib=int(row[1]),pid=os.getpid(),required_free_gib=32,policy='User-authorized node11 GPU7 only; GPU4 forbidden; preserve other jobs')
    import torch
    torch.set_num_threads(2);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('EVAL_MEMORY_FRACTION','.10')),0)
    save(WORK/'devices'/f'{os.getpid()}.json',d)
    return d

def pixels_video(path,frames,fps,size=512):
    from vico_avatarforcing import export_rgb
    export_rgb(path,frames,size,fps,18)

def publish(method,s,files,**kw):
    r=dict(id=s['id'],key=s['key'],model=method,protocol_sha256=protocol_sha(),files={str(p):sha(p) for p in files},input_image_sha256=s['reference_sha256'],input_audio_sha256=s['audio_sha256'],source_only=True,listener_audio_generated=False,at=time.time(),**kw)
    save(folder(method,s)/'complete.json',r);return r

def locked_jobs(method,scope=SCOPE,limit=None,dataset=None):
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


_last_cache_check=0.
def resident_checkpoint():
    """Release only unused allocator blocks under pressure; retain every live model."""
    global _last_cache_check
    now=time.monotonic()
    if now-_last_cache_check<5:return
    _last_cache_check=now
    import torch
    if not torch.cuda.is_initialized():return
    policy=read(WORK/'RESOURCE_POLICY.json');minimum=policy['reserve_mib']['7']*1024*1024
    free,_=torch.cuda.mem_get_info()
    if free<minimum+16*1024**3:torch.cuda.empty_cache()
