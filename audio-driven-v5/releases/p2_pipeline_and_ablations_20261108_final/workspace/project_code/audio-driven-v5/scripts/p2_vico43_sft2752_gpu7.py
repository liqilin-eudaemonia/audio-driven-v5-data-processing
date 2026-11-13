"""Resume the fixed ViCo experiment on reserved GPU7, away from training.

Keep original sampling and decoding operations. Separate their lifetimes so
the transformer and video decoder do not occupy device memory together.
"""
import argparse
import fcntl
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import p2_vico43_combined8000_sft2752 as original

WORK=original.WORK
SELF=Path(__file__).resolve()
read,save,sha=original.read,original.save,original.sha


def worker(stage,fraction):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    legacy=original.legacy;legacy.ARM=original.ARM
    namespace=dict(legacy.__dict__)
    if stage=='latent':
        def latent_only(work,items,p,rt,wr):
            wr.render(work,items,0,p,p['arms'])
        namespace['render_export']=latent_only
    else:
        source=inspect.getsource(legacy.render_export)
        needle='wr.render(work,items,0,p,p[\'arms\'])'
        assert source.count(needle)==1
        source=source.replace(needle,'None # All committed latents are decoded above; no DiT is loaded')
        exec(compile(source,str(SELF)+':decode_only','exec'),namespace)
    source=inspect.getsource(legacy.worker)
    exec(compile(source,str(SELF)+':legacy_worker','exec'),namespace)
    with patch.object(original.combined,'combined_runtime',original.sft_runtime):
        namespace['worker'](WORK,'render')
    save(WORK/(stage.upper()+'_GPU7_COMPLETE.json'),dict(passed=True,at=time.time(),fraction=fraction))


def run():
    p=original.prepare(WORK)
    assert p['render_gpus']==[7]
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'RUNTIME.json',dict(pid=os.getpid(),host=os.uname().nodename,at=time.time(),driver=str(SELF),gpu=7))
        if not (WORK/'LATENT_GPU7_COMPLETE.json').exists():
            for attempt,(required,fraction) in enumerate([(60000,.40),(88000,.60)],1):
                while True:
                    free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=required:break
                    save(WORK/'status.json',dict(stage='waiting_reserved_gpu7_memory',free_mib=free,required_mib=required,total=43,at=time.time()))
                    time.sleep(10)
                save(WORK/'status.json',dict(stage='latent_gpu7',attempt=attempt,total=43,at=time.time()))
                log=WORK/f'gpu7_latent_attempt{attempt}.log'
                with log.open('w') as out:
                    result=subprocess.run([str(original.PY),str(SELF),'latent','--fraction',str(fraction)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                if result.returncode==0:break
                if attempt==1 and 'out of memory' in log.read_text(errors='replace').lower():continue
                raise RuntimeError(f'GPU7 latent attempt failed; see {log}')
        assert (WORK/'LATENT_GPU7_COMPLETE.json').exists()
        for stage,marker in [('decode','DECODE_GPU7_COMPLETE.json')]:
            if (WORK/marker).exists():continue
            save(WORK/'status.json',dict(stage=stage+'_gpu7',total=43,at=time.time()))
            with (WORK/f'gpu7_{stage}.log').open('w') as out:
                subprocess.run([str(original.PY),str(SELF),stage,'--fraction','.25'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
        assert len(list(WORK.glob('conversion/*.verified.json')))==43
        save(WORK/'GENERATION_COMPLETE.json',dict(outputs=43,at=time.time()))
        save(WORK/'RENDER_COMPLETE.json',dict(at=time.time(),gpu=7))
        save(WORK/'status.json',dict(stage='official_metrics',total=43,at=time.time()))
        with (WORK/'metrics.log').open('a') as out:
            subprocess.run([str(original.legacy.MPY),str(original.SELF),'metrics','--work',str(WORK)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
        save(WORK/'GENERATION_AND_IMAGE_COMPLETE.json',dict(outputs=43,at=time.time(),results_sha256=sha(WORK/'RESULTS.json')))
        save(WORK/'status.json',dict(stage='image_complete_motion_and_review_pending',total=43,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','latent','decode']);parser.add_argument('--fraction',type=float,default=.40)
    args=parser.parse_args()
    if args.stage=='run':run()
    else:worker(args.stage,args.fraction)
