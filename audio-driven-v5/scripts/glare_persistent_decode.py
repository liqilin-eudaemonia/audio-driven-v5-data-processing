"""Persistent native process decoder with task-boundary draining."""
from concurrent.futures import wait
import threading


class DecodePool:
    def __init__(self,factory,size):
        self.executor=factory(size);self.submitted=0;self.completed=0;self.bundles=0;self.lock=threading.Lock()
    def lease(self):
        self.bundles+=1
        return DecodeLease(self)
    def metrics(self):
        with self.lock:return dict(submitted=self.submitted,completed=self.completed,bundles=self.bundles)
    def close(self):self.executor.__exit__(None,None,None)


class DecodeLease:
    def __init__(self,pool):self.pool=pool;self.futures=[]
    def __enter__(self):return self
    def submit(self,*args,**kwargs):
        future=self.pool.executor.submit(*args,**kwargs);self.futures.append(future)
        with self.pool.lock:self.pool.submitted+=1
        def finished(_):
            with self.pool.lock:self.pool.completed+=1
        future.add_done_callback(finished)
        return future
    def __exit__(self,*args):
        # The frozen worker removes its task's /dev/shm directory on exit.
        # Complete every outstanding decode before allowing that cleanup.
        wait(self.futures);self.futures.clear()
