#!/usr/bin/env python3
"""Quality first on two CPU nodes, then resume unchanged accelerated annotation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark import j_quality_pipeline as pipeline

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('worker','controller','watchdog','contract'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--slot',type=int,default=0)
    a=p.parse_args()
    if a.command=='worker':pipeline.worker(a.work,a.slot)
    elif a.command=='contract':
        contract=pipeline.runtime_contract(a.work)
        path=a.work/'contract.json'
        if path.exists() and pipeline.read(path)!=contract:raise ValueError('Frozen runtime changed')
        pipeline.write(path,contract)
    else:getattr(pipeline,a.command)(a.work)
