"""Same admission checks, memoize immutable shared JSON/code checksums per process."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen import features
from p2_listen.contracts import sha,read,atomic_json
from p2_legacy_qwen_lora import backend
cache={}
def memo(path):
    path=str(Path(path).resolve());s=Path(path).stat();key=(path,s.st_size,s.st_mtime_ns)
    if key not in cache:cache[key]=sha(path)
    return cache[key]
features.sha=memo
for arm in ['E0F','E1F']:
    c=read(Path(__file__).parent/'configs'/f'{arm}.json');start=time.time();value=backend.preflight(c)
    for (p,size,mtime),want in cache.items():
        if sha(p)!=want:raise RuntimeError('Shared checksum source changed during admission: '+p)
    atomic_json(Path(__file__).parent/f'preflight_memoized_{arm}.json',dict(success=True,samples=value['samples'],seconds=time.time()-start,memoized_files=len(cache),final_checksums_reverified=True))
    print(arm,'complete',value['samples'],time.time()-start,flush=True)
