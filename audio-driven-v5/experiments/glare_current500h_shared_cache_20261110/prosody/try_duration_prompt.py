"""Diagnostic only: supply measured duration; do not mutate cached targets."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import prepare_glare_prosody_cache as cache
from glare_prosody_audit import attach_audit
from baseline_training.features import native_glare


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0]>30*1024**3
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    opt=native_glare()
    from float.models.intensity_encoder import IntensityEncoder
    encoder=attach_audit(IntensityEncoder(opt,cuda=True))
    args=type('Args',(),dict(canary=True,extra_samples=0,limit=None))()
    jobs=cache.jobs_for(args);old=cache.base.load_protocol(cache.base.DEFAULT);rows=[]
    for job in jobs:
        wave=cache.input_wave(job,old);duration=len(wave)/16000
        prompt=(f'The audio is attached above and lasts exactly {duration:.6f} seconds. '
                f'Time zero is its beginning. All event intervals must stay inside [0.00, {duration:.6f}] seconds.\n\n'
                +encoder.prompt)
        encoder.prompt_text=encoder._build_prompt_text(prompt);encoder.audit_generation={};encoder.audit_result=None
        tick=time.time();encoder.infer_text(wave,sampling_rate=16000)
        row=dict(sid=job['sid'],duration=duration,analysis=encoder.audit_result,generation=encoder.audit_generation,
                 prompt_text=encoder.prompt_text,elapsed_s=time.time()-tick)
        rows.append(row);cache.save(cache.WORK/'DURATION_PROMPT_PROBE.json',dict(rows=rows,diagnostic_only=True))
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
