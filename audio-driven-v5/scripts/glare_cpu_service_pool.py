"""Reuse the native CPU reaction services across task bundles. No label changes."""
import threading
import time


class ServicePool:
    def __init__(self,factory,gpu,size):
        self.native=factory;self.gpu=gpu;self.size=size;self.services={};self.lock=threading.Lock()
        self.created=0;self.reused=0;self.closed=False
    def factory(self,gpu,slot):
        assert gpu==self.gpu and 0<=slot<self.size and not self.closed
        with self.lock:
            if slot not in self.services:
                self.services[slot]=Service(self,gpu,slot)
            else:self.reused+=1
            return self.services[slot]
    def metrics(self):
        with self.lock:
            return dict(created=self.created,reused=self.reused,limit=self.size,
                        calls=sum(s.calls for s in self.services.values()),
                        elapsed_s=sum(s.elapsed_s for s in self.services.values()),
                        first_call_s=sum(s.first_call_s for s in self.services.values()),
                        pids=[s.server.process.pid for s in self.services.values() if hasattr(s.server,'process')],
                        closed=self.closed)
    def close(self):
        if self.closed:return
        self.closed=True;errors=[]
        for service in self.services.values():
            try:service.server.close()
            except Exception as exc:errors.append(exc)
        if errors:raise RuntimeError('CPU service cleanup failed') from errors[0]


class Service:
    def __init__(self,pool,gpu,slot):
        self.pool=pool;self.gpu=gpu;self.slot=slot;self.lock=threading.Lock()
        self.server=pool.native(gpu,slot);pool.created+=1
        self.calls=0;self.elapsed_s=self.first_call_s=0.
    def run(self,folder):
        with self.lock:
            # A dead native server is replaced; a failed sample is still handled
            # by the existing per-input error path, never converted to success.
            process=getattr(self.server,'process',None)
            if process is not None and process.poll() is not None:
                self.server.close();self.server=self.pool.native(self.gpu,self.slot);self.pool.created+=1
            started=time.monotonic()
            try:return self.server.run(folder)
            finally:
                elapsed=time.monotonic()-started
                if not self.calls:self.first_call_s=elapsed
                self.calls+=1;self.elapsed_s+=elapsed
    def close(self):
        # Native worker closes clients at each bundle boundary. The owning GPU
        # process closes the pool after its entire queue, including stop/error.
        pass
