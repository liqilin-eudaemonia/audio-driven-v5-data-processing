"""Experimental exact-frame/batch4 selective AMP; not a production entrypoint."""
import copy
from types import MethodType

from glare_lia_execution import Encoder
from glare_lia_invariants import install


class SelectiveEncoder(Encoder):
    def __init__(self, original):
        import torch
        from float.models.encoder import Blur
        model=copy.deepcopy(original)
        model.enc.to(memory_format=torch.channels_last)
        for layer in model.enc.modules():
            if isinstance(layer,Blur):
                old=layer.forward
                def forward(self,x,fn=old):
                    with torch.autocast('cuda',enabled=False):
                        out=fn(x.float())
                    return out.to(x.dtype)
                layer.forward=MethodType(forward,layer)
        super().__init__(model,'graph')
        self._lifetime=install(model,pad=True,weights=True,no_feats=True)
        try:
            for n in (1,2,3,4):self.graph(n)
        finally:self._lifetime()

    def forward(self,x):
        with self.torch.autocast('cuda',dtype=self.torch.float16):
            value=super().forward(x)
        return value.float()

    def encode(self,ignored,frames,batch=4):
        return super().encode(self.model,frames,batch)
