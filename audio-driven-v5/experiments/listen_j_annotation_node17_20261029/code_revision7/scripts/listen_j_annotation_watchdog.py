#!/usr/bin/env python3
"""Keep the authorized node17 supervisor alive; never start duplicate owners."""
import argparse
import os
from pathlib import Path
import signal
import time

from listen_j_annotation_supervisor import ROOT, PYTHON, alive, identity, spawn
from listen_benchmark.j_annotation import acquire, read, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    args=parser.parse_args();work=args.work.resolve()
    owner=acquire(work/'watchdog.lock',False)
    if owner is None:return
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state={'pid':os.getpid(),'start_ticks':identity(os.getpid()),'started_at':time.time(),'restarts':[]}
    while not stopped:
        try:
            supervisor=read(work/'supervisor.json')
            if not alive(supervisor):
                if (work/'STOP_SUPERVISION').exists():break
                cmd=[PYTHON,str(ROOT/'scripts/listen_j_annotation_supervisor.py'),
                     '--work',str(work),'--upstream',str(args.upstream.resolve())]
                state['restarts'].append(spawn(work,'supervisor-recovered',cmd))
                state['restarts']=state['restarts'][-50:]
                time.sleep(3)
            state['error']=None
        except Exception as error:state['error']=repr(error)
        state['heartbeat']=time.time();write(work/'watchdog.json',state)
        time.sleep(15)
    state['stopped_at']=time.time();write(work/'watchdog.json',state);owner.close()


if __name__=='__main__':main()
