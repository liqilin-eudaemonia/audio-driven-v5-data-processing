"""Detach a known V7 supervisor or live reporter on the selected machine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_corpora/seamless_listen_v7_full'


def launch(node):
    reporter=node=='report'
    if not reporter:assert int(node) in (0,1,2)
    name='report' if reporter else 'node_'+node
    cmd=[sys.executable,'-u',str(ROOT/'scripts'/('report_listen_v7_bulk.py' if reporter else 'run_listen_v7_bulk.py'))]
    cmd+=['--watch'] if reporter else ['launch-node','--node',node]
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    with (OUT/'logs'/(name+'.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,cwd=str(ROOT.parent),env=env)
    record=dict(pid=p.pid,command=cmd,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (OUT/('launch_'+node+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('node',choices=['0','1','2','report'])
    launch(parser.parse_args().node)
