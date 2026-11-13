"""Bounded CPU preparation; preserve media and source claims while overlapping work."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
import os
import time


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def prepare_call(row,old_work,work,resources):
    from .j_omni import prepare_input
    start=time.monotonic();out=prepare_input(row,old_work,work,resources)
    out['_preparation_metrics']=dict(seconds=time.monotonic()-start,pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return out


def prefetch_stream(items,submit,capacity,stop,ready_first=False):
    """Drain outstanding work before returning control to a source-lock owner.

    Futures carry errors to the original per-sample handling path. Reordering is
    only among independent clips; each item stays paired with its own future.
    """
    if type(capacity) is not int or capacity<1:raise ValueError('Invalid preparation capacity')
    iterator=iter(enumerate(items));pending={};exhausted=False;in_hand=None;wait_start=None
    def fill():
        nonlocal exhausted
        while not exhausted and len(pending)<capacity and not stop.is_set():
            try:i,item=next(iterator)
            except StopIteration:exhausted=True;break
            pending[i]=(item,submit(item))
    try:
        while not stop.is_set():
            if wait_start is None:wait_start=time.monotonic()
            fill()
            if not pending:break
            if ready_first:
                done,_=wait([f for _,f in pending.values()],timeout=.5,return_when=FIRST_COMPLETED)
                if not done:continue
                i=min(i for i,(_,f) in pending.items() if f in done)
            else:i=min(pending)
            item,future=pending.pop(i)
            if ready_first:fill()
            in_hand=future
            yield item,future,time.monotonic()-wait_start
            in_hand=None;wait_start=None
    finally:
        # A caller may close early on stop/error. Never release its source lock
        # while a decoder still reads or writes that source's input records.
        unfinished=[f for _,f in pending.values()]+([in_hand] if in_hand is not None else [])
        for f in unfinished:f.cancel()
        for f in unfinished:
            try:f.result()
            except Exception:pass


class PreparationPool:
    def __init__(self,backend,workers):
        if backend not in ('thread','process') or type(workers) is not int or not 1<=workers<=16:raise ValueError('Invalid CPU preparation profile')
        self.backend=backend
        self.pool=(ThreadPoolExecutor(max_workers=workers) if backend=='thread' else
                   ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_only))
    def stream(self,rows,old_work,work,resources,capacity,stop,ready_first):
        return prefetch_stream(rows,lambda row:self.pool.submit(prepare_call,row,old_work,work,resources),capacity,stop,ready_first)
    def __enter__(self):return self
    def __exit__(self,*_):self.pool.shutdown(wait=True,cancel_futures=True)
