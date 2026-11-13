"""Early device binding and strict source checks for each owned split worker."""
from pathlib import Path
import argparse,hashlib,json,os,socket,sys
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['full','noevent'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--resume',required=True);a=p.parse_args()
    from gate import read,write,record
    ready=read(WORK/'CODE_READY.json')
    for name,expected in ready['source'].items():
        if hashlib.sha256((WORK/name).read_bytes()).hexdigest()!=expected:raise ValueError('Split source changed: '+name)
    if hashlib.sha256(Path(a.config).read_bytes()).hexdigest()!=ready['targets'][a.arm]['config_sha256']:
        raise ValueError('Split config changed')
    assert int(os.environ['WORLD_SIZE'])==(8 if a.arm=='full'else 24)
    import torch
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import runtime,importlib
    c=importlib.import_module(runtime.PACKAGES[a.arm]+'.config').load(a.config)
    rank=int(os.environ['RANK']);mask=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    write(Path(c['output'])/'rank_logs'/f'worker_{rank:02d}.json',dict(**record(),host=socket.gethostname(),
        rank=rank,world=int(os.environ['WORLD_SIZE']),physical_gpu=mask[int(os.environ['LOCAL_RANK'])],
        cuda_mask=mask,arm=a.arm,command=sys.argv))
    print(json.dumps(dict(phase='loading',arm=a.arm,rank=rank)),flush=True)
    result=runtime.load_training(a.arm).train(c,Path(a.resume))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
