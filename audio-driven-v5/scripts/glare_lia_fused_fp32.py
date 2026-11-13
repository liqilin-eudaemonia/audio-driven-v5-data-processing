"""Fused FP32 LIA kernels. Preserve native batches, frames, precision and outputs."""
import copy
import os
from pathlib import Path
import time
from types import MethodType

import torch
import triton
import triton.language as tl

from glare_lia_fast_parallel import FastEncoder


@triton.jit
def activation(X,B,Y,N:tl.constexpr,C:tl.constexpr,HW:tl.constexpr,CL:tl.constexpr,
               SLOPE:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    channel=i%C if CL else (i//HW)%C
    x=tl.load(X+i,i<N,0).to(tl.float32)
    bias=tl.load(B+channel,i<N,0).to(tl.float32)
    z=x+bias
    negative=z*tl.full((),SLOPE,tl.float32)
    out=tl.where(z>0,z,negative)*tl.full((),SCALE,tl.float32)
    tl.store(Y+i,out,i<N)


@triton.jit
def blur(X,K,Y,N:tl.constexpr,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
         OH:tl.constexpr,OW:tl.constexpr,PX:tl.constexpr,PY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    bc=i//(OH*OW);y=(i//OW)%OH-PY;x=i%OW-PX
    acc=tl.full((BLOCK,),0,tl.float32)
    for dy in tl.static_range(4):
        for dx in tl.static_range(4):
            good=(i<N)&(y+dy>=0)&(y+dy<H)&(x+dx>=0)&(x+dx<W)
            value=tl.load(X+bc*(H*W)+(y+dy)*W+x+dx,good,0).to(tl.float32)
            weight=tl.load(K+(3-dy)*4+3-dx).to(tl.float32)
            acc=tl.fma(value,weight,acc)
    tl.store(Y+i,acc,i<N)


class FusedEncoder(FastEncoder):
    def __init__(self,original,with_blur=False):
        from float.models.encoder import FusedLeakyReLU,Blur
        self.with_blur=with_blur
        model=copy.deepcopy(original)
        for layer in model.enc.modules():
            if isinstance(layer,FusedLeakyReLU):
                def forward(self,x):
                    assert x.dtype==torch.float32
                    assert x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)
                    out=torch.empty_like(x)
                    cl=not x.is_contiguous()
                    activation[(triton.cdiv(x.numel(),512),)](x,self.bias,out,x.numel(),x.shape[1],x.shape[2]*x.shape[3],cl,
                                                            self.negative_slope,self.scale,512,enable_fp_fusion=False)
                    return out
                layer.forward=MethodType(forward,layer)
            elif with_blur and isinstance(layer,Blur):
                old=layer.forward
                def forward(self,x,fn=old):
                    # Preserve native NCHW output layout and avoid an expensive
                    # transpose kernel on the largest channels-last inputs.
                    if not x.is_contiguous() or x.dtype!=torch.float32 or tuple(self.kernel.shape)!=(4,4):
                        return fn(x)
                    px,py=self.pad
                    oh=x.shape[2]+px+py-3;ow=x.shape[3]+px+py-3
                    out=torch.empty((x.shape[0],x.shape[1],oh,ow),device=x.device,dtype=x.dtype)
                    blur[(triton.cdiv(out.numel(),256),)](x,self.kernel,out,out.numel(),x.shape[1],x.shape[2],x.shape[3],
                                                       oh,ow,px,px,256,enable_fp_fusion=True)
                    return out
                layer.forward=MethodType(forward,layer)
        super().__init__(model)

    def encode(self,ignored,frames,batch=4):
        return super().encode(self.model,frames,batch)

    def profile(self):
        return {**super().profile(),'mode':'fused2','fused_activation':True,'fused_blur':self.with_blur}


def sources(pipeline):
    here=Path(__file__)
    return {name:pipeline.sha(here.with_name(name)) for name in (
        'glare_lia_fused_fp32.py','glare_lia_fast_parallel.py',
        'glare_lia_parallel.py','glare_lia_invariants.py','glare_lia_execution.py')}


def install(pipeline,mode):
    assert mode=='fused2'
    acceptance=pipeline.WORK/'gpu7_only/lia_fused_fp32_acceptance/RESULT.json'
    report=pipeline.read(acceptance)
    assert report['passed'] and report['sources']==sources(pipeline),'Fused execution has no matching acceptance'
    original=pipeline.base.encode;finish=pipeline.finish_stage
    engines={};last=[0.]
    def encode(model,frames,batch=4):
        if batch!=4:return original(model,frames,batch=batch)
        key=id(model)
        if key not in engines:engines[key]=FusedEncoder(model,True)
        engine=engines[key];value=engine.encode(model,frames,batch)
        if time.monotonic()-last[0]>=2:
            pipeline.save(pipeline.WORK/'workers/gpu7.lia_execution.json',dict(
                **engine.profile(),pid=os.getpid(),execution_source_sha256=pipeline.sha(Path(__file__)),
                sources=sources(pipeline),acceptance=str(acceptance),triton=triton.__version__,at=time.time()))
            last[0]=time.monotonic()
        return value
    def receipt(job,kind,payload,protocol,**extra):
        if kind=='lia' and extra.get('reused_parent') is False:
            extra['execution']=dict(mode=mode,dtype='float32',batch=4,
                                    sources=sources(pipeline),acceptance=str(acceptance))
        return finish(job,kind,payload,protocol,**extra)
    pipeline.base.encode=encode
    pipeline.finish_stage=receipt
