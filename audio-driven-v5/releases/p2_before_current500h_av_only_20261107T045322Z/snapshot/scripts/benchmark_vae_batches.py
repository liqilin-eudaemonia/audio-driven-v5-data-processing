"""Measure actual H200 activation memory and throughput, with resident encoders."""
import gc
from itertools import islice
from pathlib import Path
import time


def benchmark(enc,rows,output):
    import torch
    from p3_listen.common import write,read
    if Path(output).exists() and read(output).get('status')=='complete':return
    candidates=list(islice((r for r in rows if r['frames']==201 and
        (enc.root/'media'/r['split']/r['sample_id']/'complete.json').exists()),4))
    if len(candidates)!=4:raise ValueError('Need four full-length ready windows for worst-case benchmark')
    prepared=[enc.prepare(r,force=True) for r in candidates]
    values=[torch.from_numpy(x['pixels']).permute(3,0,1,2)[None].to(enc.device,torch.bfloat16)/127.5-1 for x in prepared]
    references=[];records=[]
    @torch.inference_mode()
    def encode(xs):
        x=torch.cat(xs,dim=0) if len(xs)>1 else xs[0]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=enc.ve(x);first=enc.ve(x[:,:,:1])
        z[:,:,:1]=first
        return z.cpu(),first.cpu()
    # Warm up the single-window encoder before measuring each shape.
    encode(values[:1]);torch.cuda.synchronize()
    for batch in (1,2,3,4):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        try:
            start=time.monotonic();output_z,output_ref=encode(values[:batch]);torch.cuda.synchronize();elapsed=time.monotonic()-start
            # A second iteration measures warm execution for the chosen batch shape.
            start=time.monotonic();z,ref=encode(values[:batch]);torch.cuda.synchronize();warm=time.monotonic()-start
            if batch==1:
                references=[encode([v]) for v in values]
            max_error=max(float((z[i:i+1].float()-references[i][0].float()).abs().max()) for i in range(batch))
            ref_error=max(float((ref[i:i+1].float()-references[i][1].float()).abs().max()) for i in range(batch))
            equal=all(torch.equal(z[i:i+1],references[i][0]) and torch.equal(ref[i:i+1],references[i][1]) for i in range(batch))
            records.append(dict(batch=batch,first_seconds=elapsed,warm_seconds=warm,seconds_per_window=warm/batch,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                success=True,bitwise_equal=equal,max_latent_abs_diff=max_error,max_reference_abs_diff=ref_error))
            del output_z,output_ref,z,ref
        except torch.cuda.OutOfMemoryError as e:
            records.append(dict(batch=batch,success=False,error='CUDA out of memory',detail=str(e)[:1200],
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            gc.collect();torch.cuda.empty_cache()
            break
        print({'vae_batch_benchmark':records[-1]},flush=True)
        write(output,dict(status='running',records=records,frames=201,height=512,width=512))
    del prepared,values,references;gc.collect();torch.cuda.empty_cache()
    write(output,dict(status='complete',records=records,frames=201,height=512,width=512,
        device=torch.cuda.get_device_name(),total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
        sample_ids=[r['sample_id'] for r in candidates],scope='video VAE + independent reference encoder; native and audio stages excluded'))
