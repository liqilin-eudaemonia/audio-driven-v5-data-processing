"""Experimental batched LIA; never installed into the production pipeline."""
import copy
import time
from types import MethodType

from glare_lia_execution import Encoder
from glare_lia_invariants import install


class Candidate(Encoder):
    def __init__(self, original, batch, mixed=True):
        import torch
        from float.models.encoder import Blur
        model = copy.deepcopy(original)
        model.enc.to(memory_format=torch.channels_last)
        self.mixed = mixed
        self.batch_size = batch
        if mixed:
            for layer in model.enc.modules():
                if isinstance(layer, Blur):
                    old = layer.forward
                    def forward(self, x, fn=old):
                        with torch.autocast('cuda', enabled=False):
                            out = fn(x.float())
                        return out.to(x.dtype)
                    layer.forward = MethodType(forward, layer)
        super().__init__(model, 'graph')
        self._lifetime = install(model, pad=True, weights=True, no_feats=True)
        try:
            self.graph(batch)
        finally:
            self._lifetime()

    def forward(self, x):
        t = self.torch
        # Only the CNN body uses autocast. Blur, FC head and direction are FP32.
        with t.autocast('cuda', dtype=t.float16, enabled=self.mixed):
            appearance, _, _ = self.model.enc(x, input_target=None)
        alpha = self.model.enc.fc(appearance.float())
        return t.sum(t.matmul(t.diag_embed(alpha), self.q.T), dim=1)

    def encode(self, ignored, frames, batch=4):
        import numpy as np
        t = self.torch
        n_batch = self.batch_size
        tick = time.monotonic()
        with t.inference_mode():
            pixels = t.from_numpy(np.array(frames)).cuda()
            static, graph, stored = self.graphs[n_batch]
            outputs = []
            for start in range(0, len(frames), n_batch):
                n = min(n_batch, len(frames)-start)
                static[:n].copy_(pixels[start:start+n])
                if n < n_batch:
                    static[n:].fill_(128)
                graph.replay()
                outputs.append(stored[:n].clone())
            value = t.cat(outputs).cpu()
        assert value.dtype == t.float32 and value.shape == (len(frames),512) and t.isfinite(value).all()
        self.calls += 1
        self.frames += len(frames)
        self.elapsed_s += time.monotonic()-tick
        return value
