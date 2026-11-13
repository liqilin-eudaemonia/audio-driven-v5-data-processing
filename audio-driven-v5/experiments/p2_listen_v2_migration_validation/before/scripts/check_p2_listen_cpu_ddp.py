#!/usr/bin/env python3
"""Two CPU/Gloo ranks vs one accumulated batch; no model weights/data/GPU."""
import copy
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
import torch.distributed as dist
from p2_listen.audit import fixture
from p2_listen.config import load
from p2_listen.model import ListeningExtractor
from p2_listen.training import optimization_step
from p2_listen.contracts import atomic_json


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank=dist.get_rank();world=dist.get_world_size()
    if world!=2:raise ValueError("Run with --nproc-per-node=2")
    try:
        torch.manual_seed(17)
        cfg,sample=fixture(load())
        samples=[copy.deepcopy(sample) for _ in range(4)]
        samples[1][4]["head_nod"]["known"].fill_(False)
        samples[3][4]["listener_vocalization"]["known"][:,2:]=False
        initial=ListeningExtractor(cfg,8)
        t=dict(connector_lr=.001,feature_lr=.001,event_lr=.001,clip_norm=1.)
        weights=dict(head_nod=1.,listener_vocalization=1.)
        callback=lambda u,_:(u.video-.2).square().mean()+(u.audio+.1).square().mean()
        results={}
        for mode in ("probe_only","joint_event"):
            reference=copy.deepcopy(initial)
            opt=torch.optim.AdamW(reference.optimizer_groups(t),weight_decay=0)
            wrapped=torch.nn.parallel.DistributedDataParallel(copy.deepcopy(initial),broadcast_buffers=False)
            opt2=torch.optim.AdamW(wrapped.module.optimizer_groups(t),weight_decay=0)
            for step in range(3):
                optimization_step(reference,opt,samples,callback,mode=mode,pos_weights=weights,event_weight=.05,
                    seed=11,ordinals=[step*4+i for i in range(4)],config=t)
                optimization_step(wrapped,opt2,samples[rank::2],callback,mode=mode,pos_weights=weights,event_weight=.05,
                    seed=11,ordinals=[step*4+i for i in range(rank,4,2)],config=t,world=2)
            errors={k:float((v-wrapped.module.state_dict()[k]).abs().max()) for k,v in reference.state_dict().items() if v.is_floating_point()}
            maximum=max(errors.values())
            if maximum>2e-5:raise AssertionError(sorted(errors.items(),key=lambda x:-x[1])[:5])
            results[mode]=dict(max_parameter_error=maximum,tolerance=2e-5)
        if rank==0:
            results.update(cpu_only=True,world_size=2,unequal_label_counts=True,gradient_accumulation=True)
            output=Path(__file__).resolve().parents[1]/"experiments/p2_listen_code_validation/cpu_ddp.json"
            atomic_json(output,results);print(results,flush=True)
    finally:dist.destroy_process_group()


if __name__=="__main__":main()
