"""CPU-only vLLM renderer parallelism experiment; never creates an LLM/GPU engine."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import asyncio
from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
import statistics
import sys
import time

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parents[1] / 'src'))


def compare(a, b, path='input'):
    import numpy as np
    import torch
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, Mapping):
        assert set(a) == set(b), path
        for key in a:
            if key != 'arrival_time':
                compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]')
    elif dataclasses.is_dataclass(a):
        assert type(a) == type(b), path
        for f in dataclasses.fields(a):
            compare(getattr(a, f.name), getattr(b, f.name), f'{path}.{f.name}')
    elif hasattr(a, '__dict__'):
        assert type(a) == type(b), path
        compare(vars(a), vars(b), path)
    else:
        assert a == b, (path, a, b)


async def main():
    os.nice(10)
    import torch
    from vllm.config import VllmConfig, ModelConfig, DeviceConfig, CacheConfig
    from vllm.renderers import renderer_from_config
    from transformers import AutoProcessor
    from listen_benchmark.visual_native8 import MODEL, MM_SIZE, prepare_media
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.shared_video_inputs import prepare_shared_inputs
    from listen_benchmark.engine_input_renderer import copy_video_metadata
    torch.set_num_threads(2)
    def renderer(workers):
        model = ModelConfig(model=MODEL, dtype='bfloat16', max_model_len=49152,
            trust_remote_code=True, mm_processor_cache_gb=0, renderer_num_workers=workers,
            mm_processor_kwargs={'size': MM_SIZE}, limit_mm_per_prompt={'image': 0, 'video': 1})
        return renderer_from_config(VllmConfig(model_config=model, device_config=DeviceConfig(device='cpu'),
                                              cache_config=CacheConfig(enable_prefix_caching=True)))
    sequential, parallel = renderer(1), renderer(2)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    row = json.loads((WORK / 'jobs/naturalistic_V02_S5078_I00000281.json').read_text())['rows'][0]
    whole = prepare_media(row['source'], frame_crop=crop_original_frame)
    inputs = [item['input'][0] for item in prepare_shared_inputs(processor, whole).values()]
    baseline = [sequential.render_cmpl([copy_video_metadata(entry)])[0] for entry in inputs]
    trial = await parallel.render_cmpl_async([copy_video_metadata(entry) for entry in inputs])
    compare(baseline, trial)
    del baseline, trial
    timings = {'serial': [], 'parallel2': []}
    for repeat in range(4):
        for mode in (['serial', 'parallel2'] if repeat % 2 == 0 else ['parallel2', 'serial']):
            tick = time.perf_counter()
            if mode == 'serial':
                output = [sequential.render_cmpl([copy_video_metadata(entry)])[0] for entry in inputs]
            else:
                output = await parallel.render_cmpl_async([copy_video_metadata(entry) for entry in inputs])
            timings[mode].append(time.perf_counter() - tick)
            del output
    report = {'at': time.time(), 'sample_id': row['sample_id'], 'windows': len(inputs),
        'scope': 'CPU-only actual vLLM renderer; uncached both arms, warmed alternating repeats; no production change',
        'exact_engine_inputs_except_arrival_time': True, 'timing_s': timings,
        'prefix_caching': True,
        'median_s': {key: statistics.median(values) for key, values in timings.items()}}
    (WORK / 'renderer_parallel_cpu_benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    sequential._executor.shutdown(wait=True)
    parallel._executor.shutdown(wait=True)


if __name__ == '__main__':
    asyncio.run(main())
