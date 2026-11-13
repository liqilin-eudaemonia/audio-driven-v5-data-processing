"""Measure an unadopted precision tradeoff; never writes production payloads."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
import prepare_glare_nonprosody_cache as base
import run_glare_shared_cache as common
from glare_lia_execution import Encoder
from glare_lia_fast_parallel import SingleFastEncoder
from glare_lia_invariants import install

OUT = common.WORK / 'gpu7_only' / 'lia_amp_probe'


class AmpEncoder(Encoder):
    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.float16):
            output = super().forward(x)
        return output.float()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0] > 16 * 1024**3
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic = base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
    cohort = []
    for sid in diagnostic['diagnostic_train'] + diagnostic['diagnostic_validation']:
        cohort.append((sid, np.array(np.load(ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy', mmap_mode='r'))))
    baseline = SingleFastEncoder(model)
    restore = install(model, pad=True, weights=True, no_feats=True)
    candidate = AmpEncoder(model, 'graph')
    for n in (1, 2, 3, 4):
        candidate.graph(n)
    restore()
    from float.utils import compute_alpha_stats
    direction = model.dec.direction.weight.detach().cpu()
    errors = []
    for sid, frames in cohort:
        old = baseline.encode(model, frames)
        new = candidate.encode(model, frames)
        diff = (new-old).abs()
        a, b = compute_alpha_stats(old[1:]@direction), compute_alpha_stats(new[1:]@direction)
        errors.append(dict(sid=sid, frames=len(frames), exact=torch.equal(old,new),
                           max_error=float(diff.max()), rmse=float((diff**2).mean().sqrt()),
                           relative_l2=float(torch.linalg.vector_norm(new-old)/torch.linalg.vector_norm(old)),
                           statistics={k:dict(max_error=float((a[k]-b[k]).abs().max()), relative_l2=float(torch.linalg.vector_norm(a[k]-b[k])/torch.linalg.vector_norm(a[k]).clamp_min(1e-9))) for k in a}))
    rounds = []
    for use_new in (False, True, True, False, False, True, True, False):
        tick = time.monotonic()
        for _, frames in cohort:
            (candidate if use_new else baseline).encode(model, frames)
        torch.cuda.synchronize()
        rounds.append(dict(candidate=use_new, seconds=time.monotonic()-tick))
    old = sum(r['seconds'] for r in rounds if not r['candidate'])
    new = sum(r['seconds'] for r in rounds if r['candidate'])
    result = dict(measured=True, production_adopted=False, samples=len(cohort), frames=sum(len(f) for _,f in cohort),
                  baseline='Optimized single-stream FP32; not the original slow implementation',
                  candidate='FP16 autocast, graph, FP32 stored output', speedup=old/new,
                  errors=errors, rounds=rounds, source_sha256=base.sha(Path(__file__)),
                  warning='Nonzero numerical error changes cached training targets. Requires a new precision protocol and training-quality validation before adoption.',
                  production_cache_written=False, physical_gpu=7)
    base.save(OUT/'RESULT.json', result)
    print(json.dumps(dict(speedup=old/new, max_error=max(r['max_error'] for r in errors), relative_l2_max=max(r['relative_l2'] for r in errors), adopted=False)),flush=True)


if __name__ == '__main__':
    main()
