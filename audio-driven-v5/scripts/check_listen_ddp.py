"""Eight-rank real-model update check; does not overwrite research checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=2)
    a=p.parse_args()
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    local=int(os.environ["LOCAL_RANK"])
    # The existing construction deliberately uses cuda:0; give each rank one physical card.
    os.environ["CUDA_VISIBLE_DEVICES"]=str(local)
    os.environ["OMP_NUM_THREADS"]="2"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from p3_listen.common import config,resolve,read,write,digest,source_contract
    cfg=config()
    sys.path.insert(1,str(resolve(cfg["reuse_root"])/"src"))
    from p3_listen.training import build,checked,restore_model
    from p3_listen.features import load_native,load_av,load_text
    from omni_av.ltx_runtime import with_text
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl",device_id=torch.device("cuda:0"))
    try:
        root=resolve(cfg["output"])
        manifest=read(root/"windows.json")
        ids=read(root/"s0_selection.json")["train"]
        lookup={r["sample_id"]:r for r in manifest["windows"]}
        rows=[lookup[sid] for sid in ids]
        saved=checked(a.checkpoint)
        if saved["identity"]["code"]!=source_contract(cfg) or saved["identity"]["manifest"]!=digest(manifest):
            raise ValueError("Checkpoint source/data mismatch")
        model,strategy,sampler=build(cfg,saved["stage"],saved["identity"]["steps"])
        restore_model(model,saved)
        initial_step=saved["step"]
        del saved
        recipe=cfg["training"]
        optimizer=torch.optim.AdamW(model.optimizer_groups(recipe["connector_lr"],recipe["dit_lr"],
            recipe["planner_lr"],recipe["heads_lr"]),weight_decay=0.)
        ddp=DDP(model,device_ids=[0],broadcast_buffers=False,find_unused_parameters=True)
        model.train()
        text=load_text(root,cfg,"cuda:0")
        frozen=[(x,x._version) for m in (model.planner,model.renderer) for x in m.parameters() if not x.requires_grad]
        updates=[]
        for step in range(1,a.steps+1):
            begin=time.monotonic()
            ordinal=(step-1)*world+rank
            seq=list(range(len(rows)));random.Random(cfg["seed"]+ordinal//len(rows)).shuffle(seq)
            row=rows[seq[ordinal%len(rows)]]
            seed=cfg["seed"]+ordinal
            torch.manual_seed(seed)
            layout,targets=load_native(root,row,cfg,targets=True,device="cuda:0")
            av=load_av(root,row,cfg,"cuda:0")
            prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                losses=ddp(layout,targets,prepared,seed,initial_step+step)
            if not torch.isfinite(losses["loss"]):raise FloatingPointError("Nonfinite loss")
            losses["loss"].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            model.assert_ownership()
            if any(x._version!=v or x.grad is not None for x,v in frozen):
                raise RuntimeError("Frozen parameter mutation")
            h=hashlib.sha256()
            for name,param in model.named_parameters():
                if param.requires_grad:
                    h.update(name.encode());h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            hashes=[None]*world
            dist.all_gather_object(hashes,h.hexdigest())
            if len(set(hashes))!=1:raise RuntimeError("DDP parameters diverged")
            loss=losses["loss"].detach().float()
            dist.all_reduce(loss)
            report=dict(rank=rank,step=step,seconds=time.monotonic()-begin,
                        global_mean_loss=float(loss/world),gradient_norm=float(norm),
                        peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,parameter_sha256=h.hexdigest())
            allrows=[None]*world
            dist.all_gather_object(allrows,report)
            if rank==0:
                updates.append(allrows)
                write(a.output/"progress.json",dict(world_size=world,stage=model.stage,updates=updates))
                print(json.dumps(dict(world_size=world,step=step,loss=float(loss/world),parameters_identical=True,
                                      max_seconds=max(r["seconds"] for r in allrows))),flush=True)
        if rank==0:
            write(a.output/"completed.json",dict(world_size=world,stage=model.stage,updates=updates,
                finite_loss=True,parameters_identical=True,frozen_unchanged=True,engineering_only=True))
    finally:
        dist.destroy_process_group()


if __name__=="__main__":main()
