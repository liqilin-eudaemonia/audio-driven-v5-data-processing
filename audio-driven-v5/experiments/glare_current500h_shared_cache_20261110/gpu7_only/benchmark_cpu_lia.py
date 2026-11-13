"""CPU-only LIA capacity probe; no GPU context, production data or cache writes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                  ONEDNN_DEFAULT_FPMATH_MODE='STRICT',DNNL_DEFAULT_FPMATH_MODE='STRICT')
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'related/glare'),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch

WORK=ROOT/'experiments/glare_current500h_shared_cache_20261110'
OUT=WORK/'gpu7_only/lia_cpu_probe'
MODEL=None
Q=None


def init(threads,barrier):
    global MODEL,Q
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    from float.models.generator import Generator
    path=ROOT/'compare/AvatarForcing/pretrained_dir/flow_transformer.pth'
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state={k.removeprefix('motion_autoencoder.'):v for k,v in state.items() if k.startswith('motion_autoencoder.')}
    MODEL=Generator(size=512,style_dim=512,motion_dim=20)
    MODEL.load_state_dict(state,strict=True,assign=True)
    MODEL.eval().requires_grad_(False)
    MODEL.enc.to(memory_format=torch.channels_last)
    with torch.inference_mode():Q=MODEL.dec.direction(None).detach()
    if barrier is not None:barrier.wait(timeout=180)


def encode(task):
    sid,count=task
    path=ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy'
    frames=np.load(path,mmap_mode='r')
    if count:frames=frames[:count]
    tick=time.monotonic();values=[]
    with torch.inference_mode():
        for start in range(0,len(frames),4):
            x=torch.from_numpy(np.array(frames[start:start+4])).permute(0,3,1,2).float()/127.5-1
            appearance,_,_=MODEL.enc(x,input_target=None)
            alpha=MODEL.enc.fc(appearance)
            values.append(torch.sum(torch.matmul(torch.diag_embed(alpha),Q.T),dim=1))
        value=torch.cat(values)
    assert torch.isfinite(value).all()
    return dict(pid=os.getpid(),frames=len(value),seconds=time.monotonic()-tick,value=value)


def main():
    OUT.mkdir(exist_ok=True)
    diagnostic=json.loads((ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json').read_text())
    ids=diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']
    rows=[];parity=[];ctx=multiprocessing.get_context('spawn')
    # Fixed two threads per worker, maximum64 additional CPU threads.
    for workers in (8,16,32):
        barrier=ctx.Barrier(workers)
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init,initargs=(2,barrier)) as pool:
            warm=list(pool.map(encode,[(ids[0],4)]*workers,chunksize=1))
            if not parity:
                for sid,result in zip(ids,pool.map(encode,[(sid,0) for sid in ids])):
                    reference=torch.load(ROOT/'experiments/baseline_training_realdata_20261109/features/glare'/(sid+'.pt'),map_location='cpu',weights_only=True)['motion']
                    value=result['value'];delta=value-reference
                    parity.append(dict(sid=sid,frames=len(value),max_error=float(delta.abs().max()),relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference))))
                (OUT/'PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')
            tasks=[(ids[2],48)]*(workers*2)
            tick=time.monotonic();results=list(pool.map(encode,tasks,chunksize=1));elapsed=time.monotonic()-tick
            row=dict(workers=workers,threads_per_worker=2,frames=sum(r['frames'] for r in results),seconds=elapsed,
                     frames_per_second=sum(r['frames'] for r in results)/elapsed,unique_pids=len({r['pid'] for r in results}),
                     mean_task_s=float(np.mean([r['seconds'] for r in results])))
            rows.append(row);(OUT/'PROGRESS.json').write_text(json.dumps(dict(trials=rows,parity=parity),indent=2)+'\n')
            print(json.dumps(row),flush=True)
    (OUT/'RESULT.json').write_text(json.dumps(dict(trials=rows,parity=parity,gpu_used=False,production_adopted=False,production_cache_written=False,
                                                caution='CPU full FP32 still has device/layout rounding differences; cold source decoding not included.'),indent=2)+'\n')


if __name__=='__main__':main()
