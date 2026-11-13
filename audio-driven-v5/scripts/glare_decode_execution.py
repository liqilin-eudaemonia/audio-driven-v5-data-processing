"""Spawned CPU decoding of independent groups using the frozen frame recipe."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')


def prepare_group(group, jobs, old, tmp_root):
    # Spawn never inherits the parent's CUDA context. This subprocess only
    # reads source media and writes bounded CPU arrays for the original worker.
    import cv2
    cv2.setNumThreads(1)
    from run_glare_shared_cache import prepare_group as frozen
    return frozen(group,jobs,old,tmp_root)


class DecodeExecutor:
    def __init__(self,max_workers,observer=None):
        self.pool=ProcessPoolExecutor(max_workers=max_workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=cpu_only)
        self.observer=observer

    def submit(self,fn,group,jobs,old,tmp_root):
        # Only this group's one/two rows cross IPC, never the complete index.
        subset={o:jobs[o] for o in json.loads(group['ordinals'])}
        future=self.pool.submit(prepare_group,group,subset,old,tmp_root)
        if self.observer:
            def observed(f):
                if not f.cancelled() and f.exception() is None:
                    self.observer('prepare_group',f.result()['prepare_s'])
            future.add_done_callback(observed)
        return future

    def __enter__(self):return self

    def __exit__(self,*exc):self.pool.shutdown(wait=True,cancel_futures=True)
