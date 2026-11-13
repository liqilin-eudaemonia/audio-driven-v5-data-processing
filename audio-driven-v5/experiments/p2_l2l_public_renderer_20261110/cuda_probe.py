"""Check the public renderer's CUDA operators before model integration."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import p2_l2l_gpavatar as gp

device=gp.guard()
import torch
import nerfacc
from pytorch3d.ops import knn_points

a=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]],device='cuda')
b=torch.tensor([[[0.,0.,0.],[1.5,0.,0.],[0.,3.,0.]]],device='cuda')
actual=knn_points(a,b,K=2).dists
expected=torch.cdist(a,b).square().sort(dim=-1).values[:,:,:2]
error=float((actual-expected).abs().max())
assert error<1e-6,error
origins=torch.tensor([[0.,0.,-2.]],device='cuda')
directions=torch.tensor([[0.,0.,1.]],device='cuda')
boxes=torch.tensor([[-1.,-1.,-1.,1.,1.,1.]],device='cuda')
near,far,hits=nerfacc.ray_aabb_intersect(origins,directions,boxes)
assert hits.all() and float(near[0,0])==1. and float(far[0,0])==3.
gp.save(gp.WORK/'CUDA_PROBE.json',dict(passed=True,device=device,pytorch3d_knn_max_abs=error,
    nerfacc_version=nerfacc.__version__,torch=torch.__version__,cuda=torch.version.cuda,
    official_nerfacc_cuda_compiled_and_executed=True))
print('CUDA operators passed',flush=True)
