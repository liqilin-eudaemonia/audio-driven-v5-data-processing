"""Locate slow inference operators and measure layout/selective-precision paths.

Read-only real RGB; no production cache/protocol writes. The launcher temporarily
pauses only the current GLARE producer and always resumes it.
"""
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
import prepare_glare_nonprosody_cache as base
import run_glare_shared_cache as common
from glare_lia_execution import Encoder
from glare_lia_fast_parallel import FastEncoder, SingleFastEncoder
from glare_lia_invariants import install

OUT = common.WORK/'gpu7_only/lia_layout_probe'


class AmpEncoder(Encoder):
    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.float16):
            out = super().forward(x)
        return out.float()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 20*1024**3
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic = base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
    cohort = []
    for sid in diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']:
        path = ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy'
        cohort.append((sid, np.array(np.load(path, mmap_mode='r'))))
    live = base.read(common.WORK/'workers/gpu7.json')['pid']
    for folder in list((Path('/dev/shm')/('glare-shared-7-'+str(live))).glob('group-*'))[:32]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort) >= 8:
                break
            try:
                frames = np.array(np.load(path, mmap_mode='r'))
            except FileNotFoundError:
                continue
            cohort.append((path.parent.name, frames))
        if len(cohort) >= 8:
            break
    base.save(OUT/'COHORT.json', dict(samples=[dict(sid=sid, frames=len(f)) for sid,f in cohort]))
    baseline = FastEncoder(model)
    baseline.warm()
    refs = {sid:baseline.encode(model, f) for sid,f in cohort}
    direction = model.dec.direction.weight.detach().cpu()
    from float.utils import compute_alpha_stats
    rows = []
    for label, channels_last, mixed in [('channels_last_fp32', True, False),
                                      ('amp_fp32_blur', False, True),
                                      ('channels_last_amp_fp32_blur', True, True)]:
        clone = copy.deepcopy(model)
        if channels_last:
            clone.enc.to(memory_format=torch.channels_last)
        if mixed:
            from float.models.encoder import Blur
            for layer in clone.enc.modules():
                if isinstance(layer, Blur):
                    original = layer.forward
                    def forward(self, x, fn=original):
                        with torch.autocast('cuda', enabled=False):
                            y = fn(x.float())
                        return y.to(x.dtype)
                    layer.forward = MethodType(forward, layer)
            restore = install(clone, pad=True, weights=True, no_feats=True)
            candidate = AmpEncoder(clone, 'graph')
            for n in (1,2,3,4):
                candidate.graph(n)
            restore()
        else:
            restore = None
            candidate = SingleFastEncoder(clone)
        parity = []
        for sid,frames in cohort:
            new = candidate.encode(clone, frames)
            old = refs[sid]
            diff = (new-old).abs()
            a,b = compute_alpha_stats(old[1:]@direction),compute_alpha_stats(new[1:]@direction)
            parity.append(dict(sid=sid,frames=len(frames),exact=torch.equal(old,new),max_error=float(diff.max()),
                               rmse=float((diff**2).mean().sqrt()),relative_l2=float(torch.linalg.vector_norm(new-old)/torch.linalg.vector_norm(old)),
                               statistics_relative_l2={k:float(torch.linalg.vector_norm(a[k]-b[k])/torch.linalg.vector_norm(a[k]).clamp_min(1e-9)) for k in a}))
        rounds=[]
        for use_new in (False,True,True,False):
            tick=time.monotonic()
            for sid,frames in cohort:
                candidate.encode(clone,frames) if use_new else baseline.encode(model,frames)
            torch.cuda.synchronize()
            rounds.append(dict(candidate=use_new,seconds=time.monotonic()-tick))
        old_s=sum(r['seconds'] for r in rounds if not r['candidate'])
        new_s=sum(r['seconds'] for r in rounds if r['candidate'])
        row=dict(label=label,channels_last=channels_last,mixed_precision=mixed,rounds=rounds,
                 speedup_vs_current_fast2=old_s/new_s,all_tensors_exact=all(p['exact'] for p in parity),
                 max_error=max(p['max_error'] for p in parity),relative_l2_max=max(p['relative_l2'] for p in parity),parity=parity)
        rows.append(row)
        base.save(OUT/'PROGRESS.json',dict(trials=rows))
        print(json.dumps({k:row[k] for k in ['label','speedup_vs_current_fast2','all_tensors_exact','max_error','relative_l2_max']}),flush=True)
        del candidate,clone,restore
        gc.collect()
        torch.cuda.empty_cache()

    # Inspect eager kernels on a single batch, not the graph replay envelope.
    profile_rows=[]
    pixels=torch.from_numpy(cohort[0][1][:4].copy()).cuda().permute(0,3,1,2).float()/127.5-1
    for mixed in (False,True):
        def forward():
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=mixed):
                appearance,_,_=model.enc(pixels,input_target=None)
                return model.dec.direction(model.enc.fc(appearance))
        forward();torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            forward();torch.cuda.synchronize()
        kernels=[]
        for event in prof.key_averages(group_by_input_shape=True):
            kernels.append(dict(name=event.key,shapes=event.input_shapes,calls=event.count,
                                self_device_us=getattr(event,'self_device_time_total',0),
                                device_us=getattr(event,'device_time_total',0)))
        profile_rows.append(dict(autocast_fp16=mixed,kernels=sorted(kernels,key=lambda k:k['self_device_us'],reverse=True)[:20]))
    base.save(OUT/'PROFILE.json',dict(rows=profile_rows))
    base.save(OUT/'RESULT.json',dict(trials=rows,samples=len(cohort),frames=sum(len(f) for _,f in cohort),
                                  physical_gpu=7,production_adopted=False,production_cache_written=False,
                                  source_sha256=base.sha(Path(__file__)),own_production_worker_paused=True,
                                  other_jobs_kept_running=True))


if __name__ == '__main__':
    main()
