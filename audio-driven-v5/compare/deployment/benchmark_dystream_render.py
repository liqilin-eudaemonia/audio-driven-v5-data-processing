"""Isolated rendering optimization check; never writes benchmark predictions."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/p2_compare_responsenet_random200_20261106/dystream'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
sys.path.insert(0,str(ROOT/'compare/DyStream'));os.chdir(ROOT/'compare/DyStream')
import numpy as np
import torch
from PIL import Image
import app

torch.set_num_threads(2);torch.manual_seed(137)
app.load_visualization_model()
folder=next(p.parent for p in (WORK/'generated').glob('*/complete.json'))
motion=torch.from_numpy(np.load(folder/'motion_latent.npy')).cuda().float()
if motion.ndim==3:motion=motion.squeeze(0)
indices=np.linspace(0,len(motion)-1,32).astype(int)
motion=motion[indices].contiguous()
image=Image.open(folder/'processed_reference.png').convert('RGB')
c=app._vis_ctx
reference=c['transform'](image).unsqueeze(0).cuda()
report=dict(id=folder.name,frames=len(motion),inference_outputs_touched=False,variants={})

def quant(x):return np.clip((x.permute(0,2,3,1).cpu().numpy()+1)/2*255,0,255).astype('uint8')

with torch.no_grad():
    feat=c['face_encoder'](reference)
    def render(m):return c['face_generator'](c['flow_estimator'](motion[:1],m),feat.expand(m.shape[0],*feat.shape[1:]))
    for _ in range(3):render(motion[:1])
    torch.cuda.synchronize();start=time.perf_counter()
    baseline=torch.cat([render(motion[i:i+1]) for i in range(len(motion))])
    torch.cuda.synchronize();report['baseline_seconds']=time.perf_counter()-start
    original=quant(baseline)
    try:
        static=c['flow_estimator'](motion[:1],motion[:1]).clone()
        def captured_render():return c['face_generator'](static,feat)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):captured_render()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=captured_render()
        torch.cuda.synchronize();start=time.perf_counter();parts=[]
        for i in range(len(motion)):
            static.copy_(c['flow_estimator'](motion[:1],motion[i:i+1]));graph.replay();parts.append(result.clone())
        actual=torch.cat(parts);torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;pixels=quant(actual)
        report['variants']['cuda_graph']=dict(seconds=elapsed,speedup=report['baseline_seconds']/elapsed,
            float_max_abs=float((actual-baseline).abs().max()),
            uint8_max_abs=int(np.abs(pixels.astype(int)-original.astype(int)).max()),
            changed_pixel_values=int(np.count_nonzero(pixels!=original)))
    except Exception as e:report['variants']['cuda_graph']=dict(error=str(e))
    for batch in []:
        try:
            torch.cuda.synchronize();start=time.perf_counter()
            actual=torch.cat([render(motion[i:i+batch]) for i in range(0,len(motion),batch)])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;pixels=quant(actual)
            report['variants'][f'batch{batch}']=dict(seconds=elapsed,speedup=report['baseline_seconds']/elapsed,
                float_max_abs=float((actual-baseline).abs().max()),
                uint8_max_abs=int(np.abs(pixels.astype(int)-original.astype(int)).max()),
                changed_pixel_values=int(np.count_nonzero(pixels!=original)))
        except Exception as e:report['variants'][f'batch{batch}']=dict(error=str(e))
    report['max_gpu_memory_mib']=torch.cuda.max_memory_allocated()/1024**2
(WORK/'RENDER_GRAPH_BENCHMARK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
