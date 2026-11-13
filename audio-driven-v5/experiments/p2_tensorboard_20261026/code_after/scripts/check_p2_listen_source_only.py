#!/usr/bin/env python3
"""Run real inference with a filesystem audit hook refusing all supplied GT paths."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--observation",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--deny",type=Path,action="append",default=[])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=137)
    a=p.parse_args()
    from p2_listen.config import load,activate
    from p2_listen.contracts import read,observation,path_at,atomic_json
    c=load(a.config);activate(c)
    o=observation(read(a.observation),a.observation.resolve().parent)
    root=path_at(c["output"])
    denied=[root/"labels",root/"manifests/targets.json",*[root/"cache"/part for part in ("teacher","vae","events")],*a.deny]
    denied=[str(x.resolve()) for x in denied]
    blocked=[]
    def audit(event,args):
        if event!="open" or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        name=os.path.abspath(os.fsdecode(args[0]))
        if any(name==v or name.startswith(v+os.sep) for v in denied):
            blocked.append(name)
            raise RuntimeError("Source-only inference attempted GT access: "+name)
    sys.addaudithook(audit)
    from p2_listen.generation import generate
    result=generate(c,o,a.output,checkpoint=a.checkpoint,seed=a.seed,device=a.device)
    atomic_json(a.output/"source_only_audit.json",dict(denied_paths=denied,blocked_attempts=blocked,
        filesystem_guard_active=True,generated=result["status"]=="success",gt_access=False))
    print(result,flush=True)


if __name__=="__main__":main()
