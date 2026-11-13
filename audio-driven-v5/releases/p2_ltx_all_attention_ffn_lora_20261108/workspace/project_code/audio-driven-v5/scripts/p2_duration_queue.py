#!/usr/bin/env python3
"""Keep the proven serial 24-GPU queue; select the explicit duration entrypoint."""
from pathlib import Path
import argparse,fcntl,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_duration_execution import install_backend

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    install_backend()
    from p2_guided_chain import training_queue as queue
    original=queue.commands
    def commands(*args,**kwargs):
        cluster,rows=original(*args,**kwargs)
        for row in rows:
            index=row.index(str(ROOT/'scripts/p2_guided_chain_train.py'))
            row[index]=str(ROOT/'scripts/p2_duration_train.py')
        return cluster,rows
    queue.commands=commands
    a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'duration_controller.lock').open('a') as lock:
        while not (a.root/'STOP').exists():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:queue.supervise(a.root);return
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':main()
