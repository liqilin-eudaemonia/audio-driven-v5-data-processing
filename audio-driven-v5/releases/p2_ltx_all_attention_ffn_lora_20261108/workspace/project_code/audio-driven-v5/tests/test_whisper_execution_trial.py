import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from p2_whisper_execution_trial import group_norms,PipelinedReduction
from p2_whisper_av.execution import StreamingFP64Reduction


def worker(rank,path):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        results=[]
        for cls in [StreamingFP64Reduction,PipelinedReduction]:
            gen=torch.Generator().manual_seed(17)
            ps=[torch.nn.Parameter(torch.randn(n,generator=gen)) for n in [190000,120000,3,190001]]
            opt=torch.optim.AdamW(ps,lr=.001)
            for update in range(2):
                for i,p in enumerate(ps):p.grad=None if i==2 and rank==0 else torch.full_like(p,(rank+1)*.13+update)
                acc=cls(ps,1);acc.add();acc.finish(2)
                expected=sum(float(p.grad.double().square().sum()) for p in ps)**.5
                assert group_norms([dict(name='test',params=ps)])['test']==expected
                opt.step()
            results.append(([p.detach().clone() for p in ps],opt.state_dict()))
        assert all(torch.equal(a,b) for a,b in zip(results[0][0],results[1][0]))
        for k,v in results[0][1]['state'].items():
            for name,t in v.items():assert torch.equal(t,results[1][1]['state'][k][name])
    finally:dist.destroy_process_group()


def test_two_rank_pipeline_exact_updates_and_adam(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_empty_group():
    assert group_norms([dict(name='empty',params=[])])=={'empty':0.}
