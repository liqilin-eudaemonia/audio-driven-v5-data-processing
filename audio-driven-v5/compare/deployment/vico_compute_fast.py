"""Exact inference optimizations; no parameter, precision or sampling changes."""
from collections import OrderedDict
import math
import time
import torch


class ZeroSelfAudioEncoder:
    """Memoize only the invariant projected all-zero self-audio branch in eval mode."""
    def __init__(self, joint, max_entries=64):
        if max_entries<1:raise ValueError('max_entries must be positive')
        self.joint=joint;self.original=joint.forward;self.max_entries=max_entries
        self.cache=OrderedDict();self.hits=0;self.misses=0;self.fallbacks=0
        self.versions=self.signature()

    def signature(self):
        return tuple((p.data_ptr(),p._version,p.dtype,p.device) for p in list(self.joint.audio_encoder1.parameters())+list(self.joint.audio_projection.parameters()))

    def clear(self):
        self.cache.clear();self.hits=0;self.misses=0;self.fallbacks=0
        self.versions=self.signature()

    def __call__(self,audio1,audio2,blendshape):
        joint=self.joint
        if joint.training or joint.audio_encoder1.training or joint.audio_projection.training or torch.is_grad_enabled() or torch.is_autocast_enabled() or audio1.ndim!=2 or audio1.shape[0]!=1:
            self.fallbacks+=1;return self.original(audio1,audio2,blendshape)
        versions=self.signature()
        if versions!=self.versions:self.clear()
        if torch.count_nonzero(audio1).item()!=0:
            self.fallbacks+=1;return self.original(audio1,audio2,blendshape)
        frames=math.ceil(audio1.shape[1]/16000*25)
        key=(tuple(audio1.shape),audio1.dtype,audio1.device,frames)
        if key in self.cache:
            self.hits+=1;audio_feat1=self.cache.pop(key);self.cache[key]=audio_feat1
        else:
            self.misses+=1
            audio_feat1=joint.audio_projection(joint.audio_encoder1(audio1,frame_num=frames).last_hidden_state)
            self.cache[key]=audio_feat1
            if len(self.cache)>self.max_entries:self.cache.popitem(last=False)
        audio_feat2=joint.audio_projection(joint.audio_encoder2(audio2,frame_num=frames).last_hidden_state)
        blendshape_feat=joint.blendshape_encoder(blendshape)
        return audio_feat1,audio_feat2,blendshape_feat


def install_dualtalk(model):
    adapter=ZeroSelfAudioEncoder(model.joint_encoder)
    model.joint_encoder.forward=adapter
    return adapter


class ExactCudaGraph:
    """Experimental probe only; not installed by production adapters.

    Requires immutable eval-mode parameters throughout its lifetime.
    """
    def __init__(self, function, example, warmup=3):
        assert example.is_cuda and not torch.is_grad_enabled()
        self.function=function;self.shape=tuple(example.shape);self.dtype=example.dtype;self.device=example.device
        self.static=example.detach().clone();self.graph=torch.cuda.CUDAGraph();self.calls=0;self.fallbacks=0
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        start=time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(warmup):function(self.static)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph):self.output=function(self.static)
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-start

    def __call__(self,x):
        if torch.is_grad_enabled() or tuple(x.shape)!=self.shape or x.dtype!=self.dtype or x.device!=self.device:
            self.fallbacks+=1;return self.function(x)
        self.static.copy_(x);self.graph.replay();self.calls+=1
        # Preserve ordinary forward ownership; the next replay cannot overwrite this result.
        return self.output.clone()
