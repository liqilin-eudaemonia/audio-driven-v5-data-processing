"""Two-stream exact-batch LIA with verified invariant-work elimination."""
import os
from pathlib import Path
import time

from glare_lia_parallel import ParallelEncoder
from glare_lia_execution import Encoder


class SingleFastEncoder(Encoder):
    def __init__(self, model):
        from glare_lia_invariants import install
        super().__init__(model, 'graph')
        restore = install(model, pad=True, weights=True, no_feats=True)
        self._invariant_lifetime = restore
        try:
            for n in (1, 2, 3, 4):
                self.graph(n)
        finally:
            restore()

    def warm(self):
        pass

    def profile(self):
        return {**super().profile(), 'mode': 'fast1', 'streams': 1,
                'combined_padding': True, 'frozen_scaled_weights': True,
                'unused_feature_list_omitted': True}


class FastEncoder(ParallelEncoder):
    def __init__(self, model, streams=2):
        assert streams == 2
        super().__init__(model, streams)
        self._invariant_lifetime = None

    def warm(self):
        from glare_lia_invariants import install
        restore = install(self.model, pad=True, weights=True, no_feats=True)
        # The graphs retain addresses of these constant weights. Retain their
        # tensors even after restoring the model's original Python methods.
        self._invariant_lifetime = restore
        try:
            super().warm()
        finally:
            restore()

    def profile(self):
        return dict(**{**super().profile(), 'mode': 'fast2'},
                    combined_padding=True, frozen_scaled_weights=True,
                    unused_feature_list_omitted=True)


def sources(pipeline):
    here = Path(__file__)
    return {name: pipeline.sha(here.with_name(name)) for name in (
        'glare_lia_fast_parallel.py', 'glare_lia_parallel.py',
        'glare_lia_invariants.py', 'glare_lia_execution.py')}


def new_encoder(model, streams):
    assert streams in (1, 2)
    return SingleFastEncoder(model) if streams == 1 else FastEncoder(model)


def install(pipeline, mode):
    assert mode in ('fast1', 'fast2')
    original = pipeline.base.encode
    engines = {}
    last = [0.]

    def encode(model, frames, batch=4):
        if batch != 4:
            return original(model, frames, batch=batch)
        key = id(model)
        if key not in engines:
            engines[key] = new_encoder(model, int(mode[-1]))
        engine = engines[key]
        value = engine.encode(model, frames, batch)
        if time.monotonic() - last[0] >= 2:
            pipeline.save(pipeline.WORK / 'workers' / 'gpu7.lia_execution.json', dict(
                **engine.profile(), pid=os.getpid(),
                execution_source_sha256=pipeline.sha(Path(__file__)),
                sources=sources(pipeline), at=time.time()))
            last[0] = time.monotonic()
        return value
    pipeline.base.encode = encode
