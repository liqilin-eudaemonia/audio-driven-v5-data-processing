"""Exact native batch4 LIA execution on bounded independent CUDA streams."""
import time

from glare_lia_execution import Encoder


class StreamEncoder(Encoder):
    def graph(self, n):
        t = self.torch
        if n not in self.graphs:
            tick = time.monotonic()
            stream = t.cuda.current_stream()
            static = t.full((n, 512, 512, 3), 128, dtype=t.uint8, device='cuda')
            with t.inference_mode():
                for _ in range(3):
                    self.forward(static.permute(0, 3, 1, 2).float() / 127.5 - 1)
            t.cuda.synchronize()
            graph = t.cuda.CUDAGraph()
            # Each concurrent graph needs its own capture stream so cuBLAS
            # cannot reuse a single capture stream's workspace concurrently.
            with t.inference_mode(), t.cuda.graph(graph, stream=stream):
                x = static.permute(0, 3, 1, 2).float() / 127.5 - 1
                output = self.forward(x)
            self.graphs[n] = (static, graph, output)
            self.graph_build_s += time.monotonic() - tick
        return self.graphs[n]


class ParallelEncoder:
    def __init__(self, model, streams=2):
        import torch
        assert streams in (2, 4, 8)
        self.torch, self.model = torch, model
        self.streams = [torch.cuda.Stream() for _ in range(streams)]
        self.engines = [StreamEncoder(model, 'graph') for _ in self.streams]
        self.calls = self.frames = 0
        self.elapsed_s = 0.

    def warm(self):
        # Capture serially: CUDA capture cannot overlap another capture.
        self.torch.cuda.synchronize()
        for stream, engine in zip(self.streams, self.engines):
            with self.torch.cuda.stream(stream):
                for n in (1, 2, 3, 4):
                    engine.graph(n)
        self.torch.cuda.synchronize()

    def encode(self, model, frames, batch=4):
        import numpy as np
        t = self.torch
        assert model is self.model and batch == 4 and frames.dtype == np.uint8
        assert frames.ndim == 4 and tuple(frames.shape[1:]) == (512, 512, 3)
        # Warm before using any stream: uncaptured stream activity cannot run
        # during graph capture in another stream.
        if any(len(e.graphs) != 4 for e in self.engines):
            self.warm()
        tick = time.monotonic()
        results = []
        main = t.cuda.current_stream()
        with t.inference_mode():
            pixels = t.from_numpy(np.array(frames)).cuda()
            for stream in self.streams:
                stream.wait_stream(main)
            for index, start in enumerate(range(0, len(frames), batch)):
                stream = self.streams[index % len(self.streams)]
                engine = self.engines[index % len(self.streams)]
                n = min(batch, len(frames) - start)
                with t.cuda.stream(stream):
                    static, graph, stored = engine.graphs[n]
                    static.copy_(pixels[start:start + n])
                    graph.replay()
                    output = stored.clone()
                    output.record_stream(main)
                    results.append(output)
            for stream in self.streams:
                main.wait_stream(stream)
            motion = t.cat(results).cpu()
        assert motion.dtype == t.float32 and tuple(motion.shape) == (len(frames), 512)
        assert t.isfinite(motion).all()
        self.calls += 1
        self.frames += len(frames)
        self.elapsed_s += time.monotonic() - tick
        return motion

    def profile(self):
        return dict(mode='parallel', streams=len(self.streams), batch=4,
                    dtype='float32', calls=self.calls, frames=self.frames,
                    elapsed_s=self.elapsed_s,
                    graph_build_s=sum(e.graph_build_s for e in self.engines))


def install(pipeline, mode):
    import os
    from pathlib import Path
    assert mode in ('parallel2', 'parallel4')
    original = pipeline.base.encode
    engines = {}
    last = [0.]

    def encode(model, frames, batch=4):
        if batch != 4:
            return original(model, frames, batch=batch)
        key = id(model)
        if key not in engines:
            engines[key] = ParallelEncoder(model, int(mode[-1]))
        engine = engines[key]
        value = engine.encode(model, frames, batch)
        if time.monotonic() - last[0] >= 2:
            pipeline.save(pipeline.WORK / 'workers' / 'gpu7.lia_execution.json', dict(
                **{**engine.profile(), 'mode': mode}, pid=os.getpid(),
                execution_source_sha256=pipeline.sha(Path(__file__)),
                encoder_execution_sha256=pipeline.sha(Path(__file__).with_name('glare_lia_execution.py')),
                at=time.time()))
            last[0] = time.monotonic()
        return value
    pipeline.base.encode = encode
