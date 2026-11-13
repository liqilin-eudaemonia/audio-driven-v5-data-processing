from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
from glare_persistent_decode import DecodePool


class DecodePoolTests(unittest.TestCase):
    def test_executor_is_reused_and_task_exit_waits_for_decode(self):
        executors=[];started=threading.Event();release=threading.Event();exited=threading.Event()
        class Native:
            def __init__(self,n):self.pool=ThreadPoolExecutor(n);executors.append(self)
            def submit(self,*args,**kwargs):return self.pool.submit(*args,**kwargs)
            def __exit__(self,*args):self.pool.shutdown(wait=True)
        pool=DecodePool(Native,2)
        def run():
            with pool.lease() as lease:lease.submit(lambda:(started.set(),release.wait(3)))
            exited.set()
        worker=threading.Thread(target=run);worker.start();self.assertTrue(started.wait(2));self.assertFalse(exited.wait(.05))
        release.set();worker.join(3)
        with pool.lease() as lease:future=lease.submit(lambda:42)
        self.assertEqual(future.result(),42);self.assertEqual(len(executors),1)
        self.assertEqual(pool.metrics(),dict(submitted=2,completed=2,bundles=2));pool.close()


if __name__=='__main__':unittest.main()
