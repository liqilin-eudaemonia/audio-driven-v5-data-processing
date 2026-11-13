"""Bounded CPU-only persistence; native serialization/receipts remain unchanged."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
import threading
import time


class BoundedWriter:
    def __init__(self,capacity):
        assert capacity>0
        self.capacity=capacity;self.slots=threading.BoundedSemaphore(capacity)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='glare-write')
        self.local=threading.local();self.lock=threading.Lock();self.error=None
        self.submitted=self.finished=self.succeeded=0;self.work_s=self.backpressure_s=0.;self.closed=False
    def check(self):
        if self.error is not None:raise RuntimeError('Background cache write failed; no task completion allowed') from self.error
    def submit(self,fn,*args,**kwargs):
        self.check();tick=time.monotonic()
        while not self.slots.acquire(timeout=.25):self.check()
        with self.lock:self.backpressure_s+=time.monotonic()-tick;self.submitted+=1
        def write():
            started=time.monotonic();self.local.active=True
            try:
                self.check();value=fn(*args,**kwargs)
                with self.lock:self.succeeded+=1
                return value
            except BaseException as exc:
                with self.lock:
                    if self.error is None:self.error=exc
                raise
            finally:
                self.local.active=False
                with self.lock:self.finished+=1;self.work_s+=time.monotonic()-started
                self.slots.release()
        try:return self.pool.submit(write)
        except BaseException:
            self.slots.release();raise
    def in_writer(self):return bool(getattr(self.local,'active',False))
    def metrics(self):
        with self.lock:
            return dict(capacity=self.capacity,submitted=self.submitted,finished=self.finished,
                        pending=self.submitted-self.finished,succeeded=self.succeeded,
                        work_s=self.work_s,backpressure_s=self.backpressure_s,failed=self.error is not None)
    def close(self):
        if not self.closed:self.closed=True;self.pool.shutdown(wait=True)
        self.check()


class LiaWrites(AbstractContextManager):
    """Execution hooks around the frozen worker, including its disk dependencies.

    The next GPU pair can run while native CPU tensors/receipts are persisted.
    Statistics may read an immutable pending motion tensor from RAM. The writer's
    own torch.load always reads the actual file for the original roundtrip test.
    Completion receipts are serialized after all preceding stage writes.
    """
    def __init__(self,pipeline,capacity=16,torch_module=None):
        if torch_module is None:import torch as torch_module
        self.p=pipeline;self.t=torch_module;self.writer=BoundedWriter(capacity)
        self.pending={};self.lock=threading.Lock();self.committed=0;self.last_status=None
        self.originals=(pipeline.save_tensor,pipeline.finish_stage,pipeline.save,torch_module.load)
    def __enter__(self):
        original_tensor,original_finish,original_save,original_load=self.originals
        def tensor(path,values):
            if self.writer.in_writer():return original_tensor(path,values)
            self.writer.check();key=str(Path(path));holder=dict(values=values)
            with self.lock:self.pending[key]=holder
            try:future=self.writer.submit(original_tensor,path,values)
            except BaseException:
                with self.lock:self.pending.pop(key,None)
                raise
            def release(_):
                with self.lock:
                    if self.pending.get(key) is holder:self.pending.pop(key,None)
            future.add_done_callback(release)
        def load(path,*args,**kwargs):
            if not self.writer.in_writer():
                self.writer.check()
                with self.lock:holder=self.pending.get(str(Path(path))) if isinstance(path,(str,Path)) else None
                if holder is not None:
                    assert kwargs.get('map_location')=='cpu','Pending tensors are CPU-only'
                    return holder['values']
            return original_load(path,*args,**kwargs)
        def finish(*args,**kwargs):
            if self.writer.in_writer():return original_finish(*args,**kwargs)
            return self.writer.submit(original_finish,*args,**kwargs)
        def complete(path,value):
            original_save(path,value)
            with self.lock:self.committed+=1
        def save(path,value):
            path=Path(path)
            if self.writer.in_writer():return original_save(path,value)
            if path.name.endswith('.complete.json'):return self.writer.submit(complete,path,value)
            if path.parent.name=='workers' and path.name=='gpu%d.json'%value.get('gpu',-1):
                self.writer.check()
                self.last_status=(path,dict(value));value=self.status(value)
            return original_save(path,value)
        self.p.save_tensor=tensor;self.p.finish_stage=finish;self.p.save=save;self.t.load=load
        return self
    def status(self,state):
        value=dict(state);value['scheduled_completed']=state.get('completed',0)
        value['completed']=self.committed;value['async_writes']=self.writer.metrics()
        if value.get('stage') in ('finished','finished_with_errors') and value['async_writes']['pending']:
            value['stage']='draining_writes'
        return value
    def __exit__(self,typ,error,tb):
        write_error=None
        try:self.writer.close()
        except BaseException as exc:write_error=exc
        finally:
            original_tensor,original_finish,original_save,original_load=self.originals
            self.p.save_tensor=original_tensor;self.p.finish_stage=original_finish;self.p.save=original_save;self.t.load=original_load
            if self.last_status:
                path,state=self.last_status;state=self.status(state)
                if write_error:state.update(stage='failed',error=repr(write_error))
                original_save(path,state)
            self.pending.clear()
        if write_error:raise write_error
        return False
