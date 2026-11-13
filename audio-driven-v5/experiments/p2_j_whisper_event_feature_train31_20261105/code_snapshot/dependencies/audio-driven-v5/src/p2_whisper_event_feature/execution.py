"""Bound CPU cache prefetch; preserve the exact consumed sample/noise schedule."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
from p2_whisper_av.execution import settings as whisper_settings


def settings(c):
    base=whisper_settings(c)
    extra=dict(prefetch_batches=0,gradient_bucket_mb=base['gradient_bucket_mb'])
    supplied=c.get('combined_execution',{})
    if set(supplied)-set(extra):raise ValueError('Unknown combined execution setting')
    extra.update(supplied)
    if (type(extra['prefetch_batches']) is not int or not 0<=extra['prefetch_batches']<=4
            or type(extra['gradient_bucket_mb']) is not int or extra['gradient_bucket_mb']<1):
        raise ValueError('Invalid bounded prefetch / communication settings')
    return dict(base,**extra)


class CPUInputStream:
    """Only frozen CPU caches are prefetched; no CUDA or training RNG in worker."""
    def __init__(self,dataset,requests,depth=0):
        if type(depth) is not int or not 0<=depth<=4:raise ValueError('Invalid prefetch depth')
        self.dataset=dataset;self.requests=iter(requests);self.depth=depth;self.pending=deque();self.closed=False
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='combined-cache') if depth else None
        if self.pool:
            for _ in range(depth):self._submit()

    def _read(self,request):
        step,chosen=request;began=time.monotonic()
        samples=[self.dataset.sample(index,'cpu') for index,_ in chosen]
        return dict(step=step,chosen=chosen,samples=samples,read_seconds=time.monotonic()-began)

    def _submit(self):
        request=next(self.requests,None)
        if request is not None:self.pending.append(self.pool.submit(self._read,request))

    def __iter__(self):return self

    def __next__(self):
        if self.closed:raise StopIteration
        if not self.pool:return self._read(next(self.requests))
        if not self.pending:raise StopIteration
        future=self.pending.popleft()
        # Immediately refill, so N+1 reads overlap the caller's N GPU work.
        self._submit()
        return future.result()

    def close(self):
        if self.closed:return
        self.closed=True
        for future in self.pending:future.cancel()
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
        self.pending.clear()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
