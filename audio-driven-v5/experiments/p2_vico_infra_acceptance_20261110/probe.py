import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import sys,time,json,functools
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
import torch,numpy as np
from vico_more_common import save,read,gpu_guard


def dualtalk():
    import vico_more_dualtalk as d
    from p2_vico256_combined_eval import source_wave
    d.OUT=OUT/'dualtalk_probe';d.OUT.mkdir(exist_ok=True)
    gpu_guard(16);torch.manual_seed(137)
    m=d.construct().eval().cuda();proc=d.processor()
    samples=read(ROOT/'experiments/p2_compare_vico43_20261106/dystream/protocol.json')['samples']
    chosen=[min(samples,key=lambda x:x['frames']),sorted(samples,key=lambda x:x['frames'])[len(samples)//2],max(samples,key=lambda x:x['frames'])]
    items={s['id']:s for s in read(ROOT/'experiments/p2_vico256_combined8000_20261106/shards/00.json')}
    rows=[]
    for sample in chosen:
        uid=sample['id'];speaker=np.load(d.DECA/'features/speaker'/(uid+'.npz'))['motion']
        wave,_=source_wave(items[uid],ROOT/'experiments/p2_vico256_combined8000_20261106/conditions/vico'/uid)
        count=int(np.ceil(len(wave)/16000*25));idx=np.minimum(np.floor(np.arange(count)*30/25+.5).astype(int),len(speaker)-1)
        args=d.inputs(proc,np.zeros_like(wave),wave,speaker[idx][:,np.r_[0:50,53:56,50:53]])
        encoders=[m.joint_encoder.audio_encoder1,m.joint_encoder.audio_encoder2];orig=[e.forward for e in encoders]
        row=dict(id=uid,frames=count,attention_classes=[type(e.encoder.layers[0].attention).__name__ for e in encoders],runs=[])
        with torch.inference_mode():
            m(*args)
            reference=None
            for fast in [False,True,True,False]:
                for e,f in zip(encoders,orig):e.forward=functools.partial(f,output_attentions=False) if fast else f
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                result=m(*args);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                if reference is None:reference=result.clone()
                row['runs'].append(dict(fast=fast,seconds=elapsed,max_abs=float((result-reference).abs().max()),mse=float((result-reference).square().mean()),peak_mib=torch.cuda.max_memory_allocated()/1024**2))
            for e,f in zip(encoders,orig):e.forward=f
        rows.append(row);save(OUT/'DUALTALK_PROBE.json',rows);print(row,flush=True)


def gpavatar():
    import p2_l2l_gpavatar as gp
    gp.guard()
    original=gp.save
    gp.save=lambda p,x:original(OUT/('GP_'+Path(p).name),x)
    m=gp.renderer();p=gp.read(gp.WORK/'protocol.json');sample=p['samples'][0]
    avatar=torch.load(gp.WORK/'references'/sample['id']/'avatar.pt',map_location='cpu',weights_only=True)
    tracked=torch.load(gp.WORK/'references'/sample['id']/'tracked.pt',map_location='cpu',weights_only=False)
    avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float()
    motion=torch.from_numpy(np.load(sample['predictions']['conan']['path'])['motion'].astype(np.float32)).cuda()
    with torch.inference_mode():
        m.build_avatar(avatar)
        for _ in range(3):gp.render_motion(m,avatar,motion[:1])
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            gp.render_motion(m,avatar,motion[:1]);torch.cuda.synchronize()
    text=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30)
    (OUT/'GPAVATAR_PROFILE.txt').write_text(text)
    print(text,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dualtalk':dualtalk()
    else:gpavatar()
