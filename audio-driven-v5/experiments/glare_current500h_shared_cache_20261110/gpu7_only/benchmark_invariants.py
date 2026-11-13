"""Same FP32 operators, removing redundant copies and frozen scaling."""
import hashlib
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
from glare_lia_invariants import install

OUT = common.WORK / 'gpu7_only' / 'lia_invariants'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0] > 16 * 1024**3
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic = base.read(ROOT / 'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
    cohort = []
    for sid in diagnostic['diagnostic_train'] + diagnostic['diagnostic_validation']:
        frames = np.load(ROOT / 'experiments/baseline_training_realdata_20261109/media' / sid / 'listener.npy', mmap_mode='r')
        cohort.append((sid, np.array(frames)))
    live = base.read(common.WORK / 'workers/gpu7.json')['pid']
    shared = Path('/dev/shm') / ('glare-shared-7-' + str(live))
    for folder in list(shared.glob('group-*'))[:40]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort) >= 16:
                break
            try:
                frames = np.array(np.load(path, mmap_mode='r'))
            except FileNotFoundError:
                continue
            cohort.append((path.parent.name, frames))
        if len(cohort) >= 16:
            break
    assert len(cohort) >= 8
    base.save(OUT / 'COHORT.json', dict(samples=[dict(sid=sid, frames=len(f), sha256=hashlib.sha256(f.tobytes()).hexdigest()) for sid, f in cohort]))
    reference = {sid: base.encode(model, f) for sid, f in cohort}
    serial = Encoder(model, 'graph')
    for n in (1, 2, 3, 4):
        serial.graph(n)
    rows = []
    for options in [dict(pad=True, weights=False, no_feats=False), dict(pad=True, weights=True, no_feats=False), dict(pad=True, weights=True, no_feats=True)]:
        restore = install(model, **options)
        candidate = Encoder(model, 'graph')
        for n in (1, 2, 3, 4):
            candidate.graph(n)
        restore()
        parity = []
        for sid, frames in cohort:
            result = candidate.encode(model, frames)
            parity.append(dict(sid=sid, frames=len(frames), exact=torch.equal(result, reference[sid]), max_error=float((result-reference[sid]).abs().max())))
        rounds = []
        for use_new in (False, True, True, False):
            tick = time.monotonic()
            for sid, frames in cohort:
                (candidate if use_new else serial).encode(model, frames)
            torch.cuda.synchronize()
            rounds.append(dict(candidate=use_new, seconds=time.monotonic()-tick))
        old = sum(r['seconds'] for r in rounds if not r['candidate'])
        new = sum(r['seconds'] for r in rounds if r['candidate'])
        row = dict(options=options, all_tensors_exact=all(p['exact'] for p in parity), max_error=max(p['max_error'] for p in parity), parity=parity, rounds=rounds, speedup_vs_current_graph=old/new)
        rows.append(row)
        base.save(OUT/'PROGRESS.json', dict(trials=rows))
        print(json.dumps({k:row[k] for k in ['options','all_tensors_exact','max_error','speedup_vs_current_graph']}),flush=True)
        del candidate, restore
        torch.cuda.empty_cache()
    base.save(OUT/'RESULT.json', dict(trials=rows,samples=len(cohort),frames=sum(len(f) for _,f in cohort),source_sha256=base.sha(ROOT/'scripts/glare_lia_invariants.py'),physical_gpu=7,batch=4,dtype='float32',numerical_source_files_unchanged=True,production_cache_written=False))


if __name__ == '__main__':
    main()
