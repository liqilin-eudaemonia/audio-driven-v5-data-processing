#!/usr/bin/env python3
"""Restore a real checkpoint and replay its next update against a saved reference."""
import argparse
import copy
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def compare_forward_records(record, repeated, reference, *, atol=1e-6, rtol=1e-5):
    """Floating objectives use tolerance; discrete supervision counts stay exact."""
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError("Forward tolerances must be finite and nonnegative")
    counts_exact = record["event_known_bins"] == repeated["event_known_bins"] == reference["event_known_bins"]
    checks = {}
    for name, a, b in (("first_reference", record, reference),
                       ("repeat_reference", repeated, reference), ("first_repeat", record, repeated)):
        for key in ("base", "event"):
            x, y = a[key], b[key]
            checks[name + "." + key] = dict(actual=x, expected=y, abs_error=abs(x-y),
                passed=math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,abs_tol=atol,rel_tol=rtol))
    exact = counts_exact and all(v["actual"] == v["expected"] for v in checks.values())
    return dict(passed=counts_exact and all(v["passed"] for v in checks.values()),
                exact=exact, counts_exact=counts_exact, atol=atol, rtol=rtol, comparisons=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--expected",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model-atol",type=float,default=1e-6)
    p.add_argument("--optimizer-atol",type=float,default=1e-5)
    p.add_argument("--forward-atol",type=float,default=1e-6)
    p.add_argument("--forward-rtol",type=float,default=1e-5)
    a=p.parse_args()
    import torch
    from p2_listen.numerics import configure_training_numerics, validate_numerics
    numerics=configure_training_numerics()
    from p2_listen.config import activate,extractor_config,source_identity
    from p2_listen.contracts import atomic_json
    from p2_listen.training import (checked_checkpoint,restore_checkpoint,indices,
                                   optimization_step,MODES,rng_state)
    from p2_listen.dataset import load_observations,load_targets
    from p2_listen.features import load_training_sample,load_text,weights_identity
    from p2_listen.annotations import mask_classes
    from p2_listen.model import ListeningExtractor
    from p2_listen.runtime import av_loss
    saved=checked_checkpoint(a.checkpoint)
    ident=saved["identity"];c=ident["config"];activate(c)
    validate_numerics(ident.get("numerics"),numerics)
    if ident["world"]!=1 or ident["source"]!=source_identity(c):
        raise ValueError("Replay requires matching source and a single-rank checkpoint")
    if ident["assets"]!={k:weights_identity(c,k) for k in ("omni","ltx","gemma")}:
        raise ValueError("Pretrained assets changed")
    from omni_av_p2.ltx import load_renderer,append_conditions
    from omni_av.ltx_runtime import with_text
    device="cuda:0";torch.cuda.set_device(0)
    cfg=extractor_config(c);t=c["training"];seed=ident["seed"]
    torch.manual_seed(seed)
    model=ListeningExtractor(cfg,c["events"]["head_width"]).to(device)
    opt=torch.optim.AdamW(model.optimizer_groups(t),weight_decay=t["weight_decay"])
    step=saved["step"]+1
    renderer,strategy,sampler=load_renderer(cfg,device);renderer.train();model.train()
    text=load_text(c,device)["positive"]
    def callback(u,pack):
        prepared=strategy.prepare_training_inputs(with_text(pack["av"],text),sampler)
        return av_loss(renderer,append_conditions(prepared,u),pack["support"])
    rows=load_observations(c,"train");targets=load_targets(c,rows)
    chosen=indices(len(rows),step,t["global_batch"],seed)
    samples=[load_training_sample(c,rows[i],targets[rows[i]["sample_id"]],device) for i,_ in chosen]
    samples=[(*s[:4],mask_classes(s[4],ident["K_train"])) for s in samples]
    for group in opt.param_groups:
        key={"connector":"connector_lr","features":"feature_lr","events":"event_lr"}[group["name"]]
        group["lr"]=t[key]*min(1.,step/t["warmup_steps"])
    def comparison():
        return dict(tensor_count=0,unequal_tensors=[],max_abs_error=0.,scalar_mismatches=[])
    def compare(actual,want,path,report):
        if isinstance(want,torch.Tensor):
            actual=actual.detach().cpu();want=want.cpu();report["tensor_count"]+=1
            if actual.shape!=want.shape or actual.dtype!=want.dtype:
                raise ValueError("Tensor contract mismatch: "+path)
            if not torch.equal(actual,want):
                if not torch.isfinite(actual).all():raise ValueError("Nonfinite replay: "+path)
                error=float((actual.double()-want.double()).abs().max())
                report["max_abs_error"]=max(report["max_abs_error"],error)
                report["unequal_tensors"].append(dict(path=path,max_abs_error=error))
        elif isinstance(want,dict):
            if actual.keys()!=want.keys():raise ValueError("Key mismatch: "+path)
            for k in want:compare(actual[k],want[k],f"{path}.{k}",report)
        elif isinstance(want,(list,tuple)):
            if len(actual)!=len(want):raise ValueError("Length mismatch: "+path)
            for i,v in enumerate(want):compare(actual[i],v,f"{path}.{i}",report)
        elif actual!=want:report["scalar_mismatches"].append(path)

    def update():
        # AdamW keeps CPU step tensors by reference. Replaying twice must not
        # let the first optimizer update mutate the saved reference state.
        restore_checkpoint(dict(saved,optimizer=copy.deepcopy(saved["optimizer"])),model,opt,ident)
        restored=comparison()
        compare(model.state_dict(),saved["model"],"model",restored)
        compare(opt.state_dict(),saved["optimizer"],"optimizer",restored)
        compare(rng_state(),saved["rank_rng"][0],"rng",restored)
        if restored["unequal_tensors"] or restored["scalar_mismatches"]:
            raise RuntimeError("Checkpoint state was not restored exactly")
        for group in opt.param_groups:
            key={"connector":"connector_lr","features":"feature_lr","events":"event_lr"}[group["name"]]
            group["lr"]=t[key]*min(1.,step/t["warmup_steps"])
        return optimization_step(model,opt,samples,callback,mode=MODES[ident["arm"]],
            pos_weights={k:v["pos_weight"] for k,v in ident["statistics"].items()},
            event_weight=c["events"]["weight"],seed=seed,ordinals=[n for _,n in chosen],config=t,audit=True)

    from p2_listen.cache import cpu
    record=update()
    first=dict(model=cpu(model.state_dict()),optimizer=copy.deepcopy(cpu(opt.state_dict())),rng=cpu(rng_state()))
    repeated_record=update()
    repeated=comparison()
    for key,actual in (("model",model.state_dict()),("optimizer",opt.state_dict()),("rng",rng_state())):
        compare(actual,first[key],key,repeated)
    del first,saved
    expected=checked_checkpoint(a.expected)
    if expected["identity"]!=ident or expected["step"]!=step:
        raise ValueError("Reference must be the next update in the same run")
    report=comparison()
    compare(model.state_dict(),expected["model"],"model",report)
    compare(opt.state_dict(),expected["optimizer"],"optimizer",report)
    compare(rng_state(),expected["rank_rng"][0],"rng",report)
    def accepted(result):
        return not result["scalar_mismatches"] and all(
            x["max_abs_error"] <= (a.model_atol if x["path"].startswith("model.") else
                                   a.optimizer_atol if x["path"].startswith("optimizer.") else 0.)
            for x in result["unequal_tensors"])
    records=[__import__("json").loads(line) for line in (a.expected.parent/"train_rank0.jsonl").read_text().splitlines()]
    reference=next(v for v in records if v["step"]==step)
    forward=compare_forward_records(record,repeated_record,reference,atol=a.forward_atol,rtol=a.forward_rtol)
    report.update(step=step,checkpoint=str(a.checkpoint),expected=str(a.expected),record=record,
                  repeated_record=repeated_record,
                  state_restore_exact=True,forward_loss_exact=forward["exact"],forward_comparison=forward,repeated_update=repeated,
                  model_atol=a.model_atol,optimizer_atol=a.optimizer_atol,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),numerics=numerics,
                  passed=accepted(report) and accepted(repeated) and forward["passed"],
                  bitwise_update_match=not report["unequal_tensors"])
    atomic_json(a.output,report)
    print({k:v for k,v in report.items() if k not in ("record","unequal_tensors","repeated_update")},flush=True)
    if not report["passed"]:raise RuntimeError("Restored update exceeds numerical tolerance or state/loss differs")


if __name__=="__main__":main()
