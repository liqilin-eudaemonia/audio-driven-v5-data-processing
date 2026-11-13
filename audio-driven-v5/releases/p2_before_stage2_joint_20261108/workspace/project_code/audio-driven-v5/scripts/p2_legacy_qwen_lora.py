#!/usr/bin/env python3
"""Second-stage Qwen LoRA on the original four 50h experiments."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    from p2_matrix16.config import load
    from p2_legacy_qwen_lora import backend
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','train'])
    p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--parent')
    p.add_argument('--stop-after',type=int);p.add_argument('--audit-gradients',action='store_true')
    a=p.parse_args();c=load(a.config,formal=True)
    if a.command=='preflight':r=backend.preflight(c);r={k:v for k,v in r.items() if k not in ('source','historical_sources')}
    else:
        from p2_matrix16.training import train
        parent=a.parent or (str(Path(c['legacy_parent'])/'step_001268.pt') if not a.resume else None)
        r=train(c,parent=parent,resume_path=a.resume,stop_after=a.stop_after,audit=a.audit_gradients,backend=backend)
    print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
