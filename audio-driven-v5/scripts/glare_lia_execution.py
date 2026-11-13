"""Execution-only LIA optimization. Preserve FP32, exact batches and operators."""
import time


class Encoder:
    def __init__(self,model,mode='graph'):
        import torch
        self.torch=torch;self.model=model;self.mode=mode;self.graphs={}
        self.calls=0;self.frames=0;self.elapsed_s=0.;self.graph_build_s=0.
        assert mode in ['deferred','upload','graph']
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        # Native Direction recomputes a QR of frozen weights for each batch.
        # Keep its exact diag_embed/matmul/sum; cache only the invariant Q.
        with torch.inference_mode():self.q=model.dec.direction(None).detach()

    def forward(self,x):
        t=self.torch
        appearance,_,_=self.model.enc(x,input_target=None)
        alpha=self.model.enc.fc(appearance)
        return t.sum(t.matmul(t.diag_embed(alpha),self.q.T),dim=1)

    def graph(self,n):
        t=self.torch
        if n not in self.graphs:
            tick=time.monotonic()
            static=t.full((n,512,512,3),128,dtype=t.uint8,device='cuda')
            stream=t.cuda.Stream();stream.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(stream),t.inference_mode():
                for _ in range(3):
                    x=static.permute(0,3,1,2).float()/127.5-1
                    self.forward(x)
            t.cuda.current_stream().wait_stream(stream)
            t.cuda.synchronize()
            graph=t.cuda.CUDAGraph()
            with t.inference_mode(),t.cuda.graph(graph):
                x=static.permute(0,3,1,2).float()/127.5-1
                output=self.forward(x)
            self.graphs[n]=(static,graph,output)
            self.graph_build_s+=time.monotonic()-tick
        return self.graphs[n]

    def encode(self,model,frames,batch=4):
        import numpy as np
        t=self.torch
        assert model is self.model and batch==4 and frames.dtype==np.uint8
        assert frames.ndim==4 and tuple(frames.shape[1:])==(512,512,3)
        tick=time.monotonic();results=[]
        with t.inference_mode():
            if self.mode!='deferred':
                # Transfer compact RGB once per pair, not CPU float32 every4frames.
                pixels=t.from_numpy(np.array(frames)).cuda()
            for start in range(0,len(frames),batch):
                n=min(batch,len(frames)-start)
                if self.mode=='deferred':
                    x=t.from_numpy(np.array(frames[start:start+n])).permute(0,3,1,2).float().cuda()/127.5-1
                    appearance,_,_=model.enc(x,input_target=None)
                    output=model.dec.direction(model.enc.fc(appearance))
                elif self.mode=='upload':
                    x=pixels[start:start+n].permute(0,3,1,2).float()/127.5-1
                    output=self.forward(x)
                else:
                    static,graph,stored=self.graph(n)
                    static.copy_(pixels[start:start+n]);graph.replay()
                    # Replay reuses its output; clone before the next replay.
                    output=stored.clone()
                results.append(output)
            motion=t.cat(results).cpu()
        assert motion.dtype==t.float32 and tuple(motion.shape)==(len(frames),512)
        assert t.isfinite(motion).all()
        self.calls+=1;self.frames+=len(frames);self.elapsed_s+=time.monotonic()-tick
        return motion

    def profile(self):
        return dict(mode=self.mode,calls=self.calls,frames=self.frames,elapsed_s=self.elapsed_s,
                    graph_build_s=self.graph_build_s,captured_batch_sizes=sorted(self.graphs),
                    batch=4,dtype='float32',frozen_q_cached=self.mode!='deferred')


def install(pipeline,mode):
    original=pipeline.base.encode
    engines={};last=[0.]
    def encode(model,frames,batch=4):
        if batch!=4:return original(model,frames,batch=batch)
        key=id(model)
        if key not in engines:engines[key]=Encoder(model,mode)
        engine=engines[key]
        value=engine.encode(model,frames,batch)
        import os
        if time.monotonic()-last[0]>=2:
            pipeline.save(pipeline.WORK/'workers'/'gpu7.lia_execution.json',dict(
                pid=os.getpid(),**engine.profile(),execution_source_sha256=pipeline.sha(__import__('pathlib').Path(__file__)),at=time.time()))
            last[0]=time.monotonic()
        return value
    pipeline.base.encode=encode
