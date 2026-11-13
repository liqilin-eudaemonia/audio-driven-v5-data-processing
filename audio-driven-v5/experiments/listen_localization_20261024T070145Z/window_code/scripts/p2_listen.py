#!/usr/bin/env python3
"""Independent P2 listening CLI. CPU inspection never loads pretrained models."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    def command(name):
        q=sub.add_parser(name)
        q.add_argument("--config",type=Path,default=ROOT/"configs/p2_listen_event_base.json")
        return q
    q=command("prepare");q.add_argument("--observations",type=Path,required=True);q.add_argument("--targets",type=Path,required=True)
    q=command("import-labels");q.add_argument("--labels",type=Path,required=True)
    q=command("cache");q.add_argument("--part",choices=("input","teacher","vae","text","events","all"),required=True)
    q.add_argument("--split",choices=("train","dev","test"),default="train");q.add_argument("--device",default="cuda:0")
    q.add_argument("--shards",type=int,default=1);q.add_argument("--shard-index",type=int,default=0)
    q=command("audit");q.add_argument("--suite",choices=("config","data","cpu"),default="config");q.add_argument("--output",type=Path)
    q=command("train");q.add_argument("--arm",choices=("E0","E1"),required=True);q.add_argument("--train-seed",type=int,required=True)
    q.add_argument("--s0",action="store_true");q.add_argument("--steps",type=int);q.add_argument("--stop-after",type=int);q.add_argument("--resume",type=Path)
    q=command("generate");q.add_argument("--observation",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    q.add_argument("--checkpoint",type=Path);q.add_argument("--seed",type=int,default=137);q.add_argument("--device",default="cuda:0")
    q.add_argument("--mode",choices=("normal","text_only","remove_u","audio_swap"),default="normal");q.add_argument("--swap-audio",type=Path);q.add_argument("--release",type=Path)
    q=command("evaluate");q.add_argument("--protocol",type=Path,required=True);q.add_argument("--observations",type=Path,required=True)
    q.add_argument("--references",type=Path,required=True);q.add_argument("--predictions",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    q=command("report");q.add_argument("--results",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    return p


def main():
    a=parser().parse_args()
    from p2_listen.config import load,activate,extractor_config
    from p2_listen.contracts import read,rows,atomic_json,digest,path_at
    c=load(a.config)
    if a.command=="prepare":
        from p2_listen.dataset import prepare
        result=prepare(c,a.observations,a.targets)
    elif a.command=="import-labels":
        from p2_listen.dataset import import_labels
        result=import_labels(c,a.labels)
    elif a.command=="audit":
        if a.suite=="config":
            result=dict(config_sha256=digest(c),resolved_output=str(path_at(c["output"])),extractor=extractor_config(c).to_dict(),models_loaded=False)
        elif a.suite=="cpu":
            from p2_listen.audit import cpu_audit
            result=cpu_audit(c)
        else:
            from p2_listen.dataset import load_observations,load_targets,verify_media
            from p2_listen.contracts import audit_splits
            obs=load_observations(c);targets=load_targets(c,obs)
            for o in obs:verify_media(o,targets[o["sample_id"]])
            result=dict(counts=audit_splits(obs),verified=True)
        if a.output:atomic_json(a.output,result)
    elif a.command=="cache":
        if a.shards<1 or not 0<=a.shard_index<a.shards:raise ValueError("Invalid shard")
        from p2_listen.dataset import load_observations,load_targets
        from p2_listen.features import cache_parts
        parts=["input","teacher","text","vae","events"] if a.part=="all" else [a.part]
        # A single writer owns the shared fixed Gemma cache.
        if a.shard_index!=0 and "text" in parts:parts.remove("text")
        if a.part=="text" and a.shard_index!=0:raise ValueError("Text cache is written by shard 0 only")
        obs=[] if a.part=="text" else load_observations(c,a.split)[a.shard_index::a.shards]
        if a.part!="text" and not obs:raise ValueError("Empty cache shard")
        targets=load_targets(c,obs) if any(p in parts for p in ("teacher","vae")) else {}
        cache_parts(c,obs,targets,parts,a.device);result=dict(parts=parts,samples=len(obs))
    elif a.command=="train":
        from p2_listen.training import train
        result=train(c,a.arm,a.train_seed,s0=a.s0,steps=a.steps,resume=a.resume,stop_after=a.stop_after)
    elif a.command=="generate":
        activate(c)
        from p2_listen.generation import generate
        from p2_listen.contracts import observation
        obs=observation(read(a.observation),a.observation.resolve().parent)
        result=generate(c,obs,a.output,checkpoint=a.checkpoint,seed=a.seed,mode=a.mode,swap_audio=a.swap_audio,release=a.release,device=a.device)
    elif a.command=="evaluate":
        from p2_listen.evaluation import evaluate
        result=evaluate(read(a.protocol),rows(a.observations),rows(a.references),prediction_rows(a.predictions))
        atomic_json(a.output,result)
        result={"output":str(a.output),"mean_delta_macro_f1":result["mean_delta_macro_f1"]}
    else:
        from p2_listen.evaluation import report
        report(read(a.results),a.output);result=dict(output=str(a.output))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def prediction_rows(path):
    # Multiple arms/seeds legitimately repeat sample_id; generic rows() rejects these.
    if path.suffix==".jsonl":return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return json.loads(path.read_text())


if __name__=="__main__":main()
