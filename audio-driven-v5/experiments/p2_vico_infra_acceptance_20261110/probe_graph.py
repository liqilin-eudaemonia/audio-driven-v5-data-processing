import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import sys,time,copy
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
import torch,numpy as np
import p2_l2l_gpavatar as gp
from vico_compute_fast import ExactCudaGraph
from vico_more_common import save

gp.guard();original=gp.save;gp.save=lambda p,x:original(OUT/('GP_'+Path(p).name),x)
m=gp.renderer();sample=gp.read(gp.WORK/'protocol.json')['samples'][0]
avatar=torch.load(gp.WORK/'references'/sample['id']/'avatar.pt',map_location='cpu',weights_only=True)
tracked=torch.load(gp.WORK/'references'/sample['id']/'tracked.pt',map_location='cpu',weights_only=False)
avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float()
motion=torch.from_numpy(np.load(sample['predictions']['conan']['path'])['motion'].astype(np.float32)).cuda()
with torch.inference_mode():
    m.build_avatar(avatar)
    captured=[]
    hook=m.up_renderer.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
    reference=gp.render_motion(m,avatar,motion[:1]).clone();hook.remove()
    original_forward=m.up_renderer.forward
    graph=ExactCudaGraph(original_forward,captured[0]);m.up_renderer.forward=graph
    del captured
    fast=gp.render_motion(m,avatar,motion[:1]);delta=float((fast-reference).abs().max())
    print('Graph setup',graph.setup_seconds,'max_abs',delta,flush=True)
    runs=[]
    for mode in [False,True,True,False,False,True]:
        m.up_renderer.forward=graph if mode else original_forward
        torch.cuda.synchronize();start=time.perf_counter()
        for i in range(8):result=gp.render_motion(m,avatar,motion[i:i+1])
        torch.cuda.synchronize()
        runs.append(dict(fast=mode,seconds=time.perf_counter()-start))
    m.up_renderer.forward=original_forward
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        gp.render_motion(m,avatar,motion[:1]);torch.cuda.synchronize()
    (OUT/'GPAVATAR_PROFILE.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
save(OUT/'GPAVATAR_GRAPH_PROBE.json',dict(max_abs=delta,setup_seconds=graph.setup_seconds,runs=runs))
print(runs,flush=True)
