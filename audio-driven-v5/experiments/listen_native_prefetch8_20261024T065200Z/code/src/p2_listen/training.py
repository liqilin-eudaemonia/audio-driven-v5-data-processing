"""Matched E0/E1 updates; event normalization spans the entire optimizer step."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from .contracts import EVENTS, atomic_json, digest, path_at, read, sha
from .cache import atomic_torch, cpu
from .events import normalized_event_loss

MODES = {"E0":"probe_only", "E1":"joint_event"}


def indices(size, step, batch, seed, rank=0, world=1):
    result=[]
    if size<1 or batch%world:
        raise ValueError("Invalid dataset/global batch/world size")
    for micro in range(batch//world):
        ordinal=(step-1)*batch+micro*world+rank
        epoch,offset=divmod(ordinal,size)
        order=list(range(size)); random.Random(seed+epoch).shuffle(order)
        result.append((order[offset],ordinal))
    return result


def clip_groups(model, maximum):
    raw = model.module if hasattr(model,"module") else model
    return dict(main=float(torch.nn.utils.clip_grad_norm_(raw.extractor.parameters(),maximum,error_if_nonfinite=True)),
                events=float(torch.nn.utils.clip_grad_norm_(raw.event_heads.parameters(),maximum,error_if_nonfinite=True)))


def optimization_step(model, optimizer, samples, av_callback, *, mode, pos_weights,
                      event_weight, seed, ordinals, config, world=1, audit=False):
    """samples=(condition, layout, teacher, av, event_targets); callback runs frozen DiT."""
    device = next(model.parameters()).device
    counts = {k:torch.tensor(sum(int(s[4][k]["known"].sum()) for s in samples),device=device,dtype=torch.float32) for k in EVENTS}
    if world>1:
        for value in counts.values(): dist.all_reduce(value)
    optimizer.zero_grad(set_to_none=True)
    totals={"base":0.,"event":0.}
    for i,(sample,ordinal) in enumerate(zip(samples,ordinals)):
        condition,layout,teacher,av,labels=sample
        torch.manual_seed(seed+ordinal)
        generator=torch.Generator(device=device).manual_seed(seed+1_000_000+ordinal)
        sync = model.no_sync() if world>1 and i<len(samples)-1 else nullcontext()
        with sync:
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                values=model(condition,layout,teacher,labels,mode,pos_weights,generator)
                lav=av_callback(values["u"],av)
                base=lav
                for mod in ("video","audio"):
                    base=base+values["features"][mod]
                event=normalized_event_loss(values["event_sums"],counts,world)
                loss=base/len(samples)+event_weight*event
            if audit and i==0:
                if world!=1:raise ValueError("Run real gradient-path S0 on one rank before DDP")
                trace={}
                variables=[values["u"].video,values["u"].audio]
                for name,objective in (("av",lav),("feature",sum(values["features"].values())),("event",event)):
                    grads=torch.autograd.grad(objective,variables,retain_graph=True,allow_unused=True)
                    trace[name]=[0. if g is None else float(g.detach().float().norm()) for g in grads]
                if mode=="probe_only" and any(trace["event"]):raise RuntimeError("Probe gradient leaked into U")
                if not all(x>0 for x in trace["av"]):raise RuntimeError("AV gradient does not reach both U modalities")
                if mode=="joint_event":
                    for j,k in enumerate(("head_nod","listener_vocalization")):
                        if labels[k]["known"].any() and trace["event"][j]<=0:raise RuntimeError("Known event gradient missing")
                totals["paths_first_micro"]=trace
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training objective")
            loss.backward()
        totals["base"]+=float(base.detach())/len(samples)
        totals["event"]+=float(event.detach())
    grads=clip_groups(model,config["clip_norm"])
    optimizer.step()
    totals["grad_norm"]=grads
    return totals


def rng_state():
    # CPU audits must not initialize any CUDA contexts merely to capture RNG.
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(value):
    torch.set_rng_state(value["cpu"])
    if value["cuda"] is not None: torch.cuda.set_rng_state(value["cuda"])


def save_checkpoint(path, model, optimizer, step, identity, rank_rng):
    raw=model.module if hasattr(model,"module") else model
    atomic_torch(path,dict(model=cpu(raw.state_dict()),optimizer=cpu(optimizer.state_dict()),step=step,
                          identity=identity,rank_rng=rank_rng))
    atomic_json(Path(path).with_suffix(".complete.json"),dict(sha256=sha(path),step=step))


def checked_checkpoint(path):
    if sha(path)!=read(Path(path).with_suffix(".complete.json"))["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return torch.load(path,map_location="cpu",weights_only=True)


def restore_checkpoint(saved, model, optimizer, identity, rank=0):
    if saved["identity"]!=identity: raise ValueError("Strict resume identity mismatch")
    current=model.state_dict()
    if set(current)!=set(saved["model"]) or any(v.shape!=current[k].shape or not torch.isfinite(v).all() for k,v in saved["model"].items()):
        raise ValueError("Invalid checkpoint tensors")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rank_rng"][rank])
    return saved["step"]


def train(c, arm, seed, *, s0=False, steps=None, resume=None, stop_after=None):
    from .config import extractor_config,source_identity
    from .dataset import load_observations,load_targets
    from .features import load_training_sample,load_text,weights_identity
    from .annotations import class_statistics
    from .model import ListeningExtractor
    from .runtime import av_loss
    cfg=extractor_config(c)
    if arm not in MODES: raise ValueError("Unknown arm")
    if steps is not None and not s0: raise ValueError("Step overrides are S0 only")
    if steps is not None and steps<1: raise ValueError("Invalid step count")
    if stop_after is not None and stop_after<1: raise ValueError("Invalid stop-after")
    budget=(steps or 5) if s0 else c["training"]["steps"]
    if budget<5 and s0: raise ValueError("S0 needs at least 5 updates")
    rows=load_observations(c,"train")
    if not rows: raise ValueError("No admitted training observations")
    targets=load_targets(c,rows)
    # Fail on missing data/caches before allocating any model or starting NCCL.
    labels=[]; signatures=[]
    for row in rows:
        sample=load_training_sample(c,row,targets[row["sample_id"]])
        labels.append(sample[4]); del sample
        signatures.append(digest(read(path_at(c["output"])/"labels/canonical"/(row["sample_id"]+".json"))))
    stats=class_statistics(labels,c["events"]["pos_weight_cap"])
    pos={k:stats[k]["pos_weight"] for k in EVENTS}
    rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); local=int(os.environ.get("LOCAL_RANK",0))
    t=c["training"]
    if t["global_batch"]%world: raise ValueError("world size must divide global batch")
    if not torch.cuda.is_available(): raise RuntimeError("Real LTX training requires CUDA; use CPU tests for synthetic checks")
    torch.cuda.set_device(local); device=f"cuda:{local}"
    if world>1: dist.init_process_group("nccl")
    try:
        from omni_av_p2.ltx import load_renderer,append_conditions
        from omni_av.ltx_runtime import with_text
        torch.manual_seed(seed)
        raw=ListeningExtractor(cfg,c["events"]["head_width"]).to(device)
        initial=digest({k:sha_tensor(v) for k,v in raw.state_dict().items()})
        ident=dict(config=c,arm=arm,seed=seed,s0=s0,budget=budget,world=world,observations=digest(rows),
                   targets=digest(targets),labels=signatures,statistics=stats,initialization=initial,source=source_identity(c),
                   assets={k:weights_identity(c,k) for k in ("omni","ltx","gemma")})
        root=path_at(c["output"])
        pair=dict(config=c,seed=seed,s0=s0,budget=budget,world=world,observations=digest(rows),targets=digest(targets),
                  labels=signatures,statistics=stats,initialization=initial,source=ident["source"],assets=ident["assets"])
        out=root/"runs"/(("s0_" if s0 else "")+str(seed))/arm
        if world>1:
            identities=[None]*world;dist.all_gather_object(identities,digest(ident))
            if len(set(identities))!=1:raise ValueError("Ranks disagree on config/data/source/weights")
        error=None
        if rank==0:
            try:
                atomic_json(out.parent/"pair.json",pair,immutable=True)
                if (out/"run.json").exists() and not resume: raise FileExistsError("Run exists; use --resume or a new output")
                if resume:
                    saved=checked_checkpoint(resume)
                    if Path(resume).resolve().parent!=out.resolve():raise ValueError("Resume must belong to this run")
                    if (out/"status.json").exists() and read(out/"status.json")["step"]>saved["step"]:
                        raise ValueError("Refusing to append a rollback branch to existing logs")
                atomic_json(out/"run.json",ident,immutable=True)
            except Exception as e:error=f"{type(e).__name__}: {e}"
        if world>1:
            errors=[error];dist.broadcast_object_list(errors,src=0);error=errors[0]
        if error:raise ValueError(error)
        opt=torch.optim.AdamW(raw.optimizer_groups(t),weight_decay=t["weight_decay"])
        start=0
        if resume: start=restore_checkpoint(checked_checkpoint(resume),raw,opt,ident,rank)
        renderer,strategy,sampler=load_renderer(cfg,device)
        renderer.train()
        frozen=[(p,p._version) for p in renderer.parameters()]
        text=load_text(c,device)["positive"]
        def av_callback(u,pack):
            prepared=strategy.prepare_training_inputs(with_text(pack["av"],text),sampler)
            return av_loss(renderer,append_conditions(prepared,u),pack["support"])
        model=raw if world==1 else torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local],broadcast_buffers=False)
        model.train()
        def checkpoint(step):
            states=[rng_state()]
            if world>1:
                states=[None]*world;dist.all_gather_object(states,rng_state())
            if rank==0: save_checkpoint(out/f"step_{step:06d}.pt",raw,opt,step,ident,states)
        last=start
        for step in range(start+1,budget+1):
            stop=torch.tensor(int((out/"STOP").exists() or stop_after is not None and step>stop_after),device=device)
            if world>1: dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if stop.item(): break
            for group in opt.param_groups:
                key={"connector":"connector_lr","features":"feature_lr","events":"event_lr"}[group["name"]]
                group["lr"]=t[key]*min(1.,step/t["warmup_steps"])
            chosen=indices(len(rows),step,t["global_batch"],seed,rank,world)
            samples=[load_training_sample(c,rows[i],targets[rows[i]["sample_id"]],device) for i,_ in chosen]
            begin=time.monotonic()
            record=optimization_step(model,opt,samples,av_callback,mode=MODES[arm],pos_weights=pos,
                event_weight=c["events"]["weight"],seed=seed,ordinals=[n for _,n in chosen],config=t,world=world,audit=s0 and world==1)
            if any(p.requires_grad or p.grad is not None or p._version!=v for p,v in frozen):
                raise RuntimeError("Frozen renderer changed")
            record.update(step=step,rank=rank,seconds=time.monotonic()-begin,ordinals=[n for _,n in chosen])
            with (out/f"train_rank{rank}.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
            if rank==0: print(record,flush=True)
            last=step
            if step%t["save_every"]==0: checkpoint(step)
        checkpoint(last)
        if rank==0: atomic_json(out/"status.json",dict(step=last,budget=budget,s0=s0,quality_evaluated=False,stopped=last<budget))
        return dict(step=last,output=str(out))
    finally:
        if world>1 and dist.is_initialized(): dist.destroy_process_group()


def sha_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
