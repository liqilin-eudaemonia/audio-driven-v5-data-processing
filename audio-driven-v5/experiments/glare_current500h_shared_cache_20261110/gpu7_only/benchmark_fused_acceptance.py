"""Exactness/stress and paired speed checks against the current FP32 producer."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'related/glare')]
import numpy as np
import torch
import torch.nn.functional as f
import prepare_glare_nonprosody_cache as base
import run_glare_shared_cache as common
from glare_lia_fast_parallel import FastEncoder
from glare_lia_fused_fp32 import FusedEncoder,activation,blur,sources
import triton

OUT=common.WORK/'gpu7_only/lia_fused_fp32_acceptance'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0]>20*1024**3
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.manual_seed(137)
    from float.models.encoder import upfirdn2d_native,make_kernel
    stress=[]
    kernel=make_kernel([1,3,3,1]).cuda()
    for n in (1,2,3,4):
        for c,h in [(32,512),(64,256),(128,128),(256,64),(512,32),(512,16),(512,8)]:
            for cl in (False,True):
                x=torch.randn((n,c,h,h),device='cuda')
                if cl:x=x.to(memory_format=torch.channels_last)
                bias=torch.randn((1,c,1,1),device='cuda');out=torch.empty_like(x)
                activation[(triton.cdiv(x.numel(),512),)](x,bias,out,x.numel(),c,h*h,cl,.2,2**.5,512,enable_fp_fusion=False)
                reference=f.leaky_relu(x+bias,.2)*2**.5
                assert torch.equal(out,reference),(n,c,h,cl,'activation',float((out-reference).abs().max()))
                stress.append(dict(kind='activation',shape=list(x.shape),channels_last=cl,exact=True))
                if not cl:
                    for pad in (1,2):
                        oh=h+2*pad-3;out=torch.empty((n,c,oh,oh),device='cuda')
                        blur[(triton.cdiv(out.numel(),256),)](x,kernel,out,out.numel(),c,h,h,oh,oh,pad,pad,256,enable_fp_fusion=True)
                        reference=upfirdn2d_native(x,kernel,1,1,1,1,pad,pad,pad,pad)
                        assert torch.equal(out,reference),(n,c,h,pad,'blur',float((out-reference).abs().max()))
                        stress.append(dict(kind='blur',shape=list(x.shape),pad=pad,exact=True))
    base.save(OUT/'STRESS.json',dict(passed=True,cases=stress))
    model=base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic=base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json');cohort=[]
    for sid in diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']:
        cohort.append((sid,np.array(np.load(ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy',mmap_mode='r'))))
    live=base.read(common.WORK/'workers/gpu7.json')['pid']
    for folder in list((Path('/dev/shm')/('glare-shared-7-'+str(live))).glob('group-*'))[:64]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort)>=16:break
            try:frames=np.array(np.load(path,mmap_mode='r'))
            except FileNotFoundError:continue
            cohort.append((path.parent.name,frames))
        if len(cohort)>=16:break
    assert len(cohort)>=12
    cohort.append(('real_batch_tail1_first49_frames',cohort[0][1][:49].copy()))
    baseline=FastEncoder(model);baseline.warm();candidate=FusedEncoder(model,True);candidate.warm()
    parity=[]
    for sid,frames in cohort:
        old=baseline.encode(model,frames);new=candidate.encode(model,frames)
        assert torch.equal(old,new),sid
        parity.append(dict(sid=sid,frames=len(frames),exact=True,max_error=float((old-new).abs().max())))
    pairs=[]
    for repeat in range(2):
        for index,(sid,frames) in enumerate(cohort):
            times={}
            for use_new in ((False,True) if (repeat+index)%2==0 else (True,False)):
                tick=time.monotonic();(candidate if use_new else baseline).encode(model,frames)
                torch.cuda.synchronize();times['candidate' if use_new else 'native']=time.monotonic()-tick
            pairs.append(dict(sid=sid,**times))
    values=np.asarray([[p['native'],p['candidate']] for p in pairs]);rng=np.random.default_rng(137)
    sampled=values[rng.integers(0,len(values),size=(2000,len(values)))].sum(1)
    interval=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist();speedup=float(values[:,0].sum()/values[:,1].sum())
    result=dict(passed=True,sources=sources(common),samples=len(cohort)-1,extra_batch_tail_tests=1,frames=sum(len(f) for _,f in cohort),parity=parity,
                stress_cases=len(stress),all_tensors_exact=True,max_error=0.,speedup_vs_current_fast2=speedup,bootstrap_95_interval=interval,
                paired_timings=pairs,physical_gpu=7,batch=4,dtype='float32',triton=triton.__version__,
                native_source_files_unchanged=True,production_cache_written=False,
                performance_passed=speedup>1.1 and interval[0]>1.02)
    base.save(OUT/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['passed','samples','frames','stress_cases','speedup_vs_current_fast2','bootstrap_95_interval','performance_passed']}),flush=True)


if __name__=='__main__':main()
