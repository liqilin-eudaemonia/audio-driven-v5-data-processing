"""Paired FP32 fused-kernel probe, with exact equality admission."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import prepare_glare_nonprosody_cache as base
import run_glare_shared_cache as common
from glare_lia_fast_parallel import FastEncoder
from fused_fp32 import Candidate

OUT=common.WORK/'gpu7_only/lia_fused_fp32_probe'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0]>20*1024**3
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic=base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json');cohort=[]
    for sid in diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']:
        cohort.append((sid,np.array(np.load(ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy',mmap_mode='r'))))
    live=base.read(common.WORK/'workers/gpu7.json')['pid']
    for folder in list((Path('/dev/shm')/('glare-shared-7-'+str(live))).glob('group-*'))[:32]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort)>=8:break
            try:frames=np.array(np.load(path,mmap_mode='r'))
            except FileNotFoundError:continue
            cohort.append((path.parent.name,frames))
        if len(cohort)>=8:break
    base.save(OUT/'COHORT.json',dict(samples=[dict(sid=sid,frames=len(f)) for sid,f in cohort]))
    baseline=FastEncoder(model);baseline.warm();references={sid:baseline.encode(model,f) for sid,f in cohort}
    rows=[]
    for with_blur in (False,True):
        candidate=None
        try:
            candidate=Candidate(model,with_blur);candidate.warm()
            parity=[]
            for sid,frames in cohort:
                new=candidate.encode(model,frames);old=references[sid]
                parity.append(dict(sid=sid,frames=len(frames),exact=torch.equal(old,new),max_error=float((old-new).abs().max())))
            timings=[]
            for repeat in range(2):
                for index,(sid,frames) in enumerate(cohort):
                    times={}
                    for use_new in ((False,True) if (repeat+index)%2==0 else (True,False)):
                        tick=time.monotonic();(candidate if use_new else baseline).encode(model,frames)
                        torch.cuda.synchronize();times['candidate' if use_new else 'native']=time.monotonic()-tick
                    timings.append(dict(sid=sid,**times))
            pairs=np.asarray([[t['native'],t['candidate']] for t in timings]);rng=np.random.default_rng(137)
            sampled=pairs[rng.integers(0,len(pairs),size=(2000,len(pairs)))].sum(1)
            rows.append(dict(with_blur=with_blur,parity=parity,all_tensors_exact=all(p['exact'] for p in parity),
                             max_error=max(p['max_error'] for p in parity),timings=timings,
                             speedup=float(pairs[:,0].sum()/pairs[:,1].sum()),
                             bootstrap_95_interval=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist()))
        except Exception as exc:rows.append(dict(with_blur=with_blur,error=repr(exc),adopted=False))
        base.save(OUT/'PROGRESS.json',dict(trials=rows));print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['timings','parity']}),flush=True)
        del candidate
        gc.collect();torch.cuda.empty_cache()
    base.save(OUT/'RESULT.json',dict(trials=rows,samples=len(cohort),frames=sum(len(f) for _,f in cohort),physical_gpu=7,
                                  production_adopted=False,production_cache_written=False,source_sha256=base.sha(Path(__file__).with_name('fused_fp32.py'))))


if __name__=='__main__':main()
