"""Full ViCo43, cold silent-feature cache each pass, unchanged official kernels."""
import sys,time
from pathlib import Path
import numpy as np
import torch
from vico_more_common import ROOT,read,save,test_samples,gpu_guard
from vico_compute_fast import install_dualtalk
import vico_more_dualtalk as d

OUT=ROOT/'experiments/p2_vico_infra_acceptance_20261110'

def main():
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico256_combined_eval import source_wave
    device=gpu_guard(16);torch.manual_seed(137)
    d.OUT=OUT/'dualtalk';model=d.construct().eval().cuda();proc=d.processor()
    records={s['id']:s for s in read(ROOT/'experiments/p2_vico256_combined8000_20261106/shards/00.json')}
    start=time.perf_counter();items=[]
    for sample in test_samples():
        uid=sample['id'];speaker=np.load(d.DECA/'features/speaker'/(uid+'.npz'))['motion']
        wave,_=source_wave(records[uid],ROOT/'experiments/p2_vico256_combined8000_20261106/conditions/vico'/uid)
        count=int(np.ceil(len(wave)/16000*25));idx=np.minimum(np.floor(np.arange(count)*30/25+.5).astype(int),len(speaker)-1)
        items.append((uid,d.inputs(proc,np.zeros_like(wave),wave,speaker[idx][:,np.r_[0:50,53:56,50:53]])))
    preprocessing=time.perf_counter()-start
    adapter=install_dualtalk(model);original=adapter.original;runs=[];references={}
    with torch.inference_mode():
        model.joint_encoder.forward=original;model(*items[0][1])
        for fast in [False,True,True,False,False,True]:
            adapter.clear();model.joint_encoder.forward=adapter if fast else original
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            results=[model(*args) for uid,args in items]
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            errors=[]
            for (uid,_),pred in zip(items,results):
                if uid not in references:references[uid]=pred.clone()
                errors.append(float((pred-references[uid]).abs().max()))
            assert max(errors)==0,errors
            row=dict(fast=fast,seconds=elapsed,max_abs=max(errors),clips=len(items),hits=adapter.hits,misses=adapter.misses,
                     peak_mib=torch.cuda.max_memory_allocated()/1024**2)
            runs.append(row);print(row,flush=True)
    save(OUT/'DUALTALK_BENCHMARK.json',dict(passed=True,device=device,runs=runs,preprocessing_seconds=preprocessing,
        scope='All 43 full-length native predictions, model forward only, every fast pass starts empty feature cache',
        excluded='Model load and common input preprocessing',seed=137))

if __name__=='__main__':main()
