#!/usr/bin/env python3
"""Run the existing GLARE generator using the exact original/new P2 training package."""
import json
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def bootstrap(args):
    def arg(name):return Path(args[args.index(name)+1]).resolve() if name in args else None
    run=arg('--run');work=arg('--work')
    if run:source=json.loads((run/'run.json').read_text())['source']
    elif work and (work/'generation_protocol.json').exists():source=json.loads((work/'generation_protocol.json').read_text())['source']
    else:source={}
    location=ROOT/'src'
    for version in ('p2_preflight_v1','p2_feature_v1'):
        release=ROOT/'releases'/version/'src'
        if str(release/'p2_listen/training.py') in source:location=release;break
    sys.path.insert(0,str(location));import p2_listen
    return location

if __name__=='__main__':
    bootstrap(sys.argv[1:])
    # The existing convenience runner spawns its original CLI. Redirect only
    # those generation children through this same source-aware bootstrap.
    import subprocess
    original=subprocess.Popen
    def popen(command,*args,**kwargs):
        if isinstance(command,list) and len(command)>2 and command[1]==str(ROOT/'scripts/p2_glare_eval.py') and command[2]=='generate':
            command=[command[0],str(Path(__file__).resolve()),*command[2:]]
        return original(command,*args,**kwargs)
    subprocess.Popen=popen
    runpy.run_path(str(ROOT/'scripts/p2_glare_eval.py'),run_name='__main__')
