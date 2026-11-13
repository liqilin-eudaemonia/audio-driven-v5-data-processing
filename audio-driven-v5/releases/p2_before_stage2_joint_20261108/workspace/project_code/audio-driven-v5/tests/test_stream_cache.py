import sys
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from listen_stream_cache import prefetched,job_lock

class StreamCacheTests(unittest.TestCase):
    def test_prefetch_complete_order_and_failure(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(prefetched(pool,lambda x:x*x,range(31))),[x*x for x in range(31)])
            def broken(x):
                if x==4:raise ValueError('invalid media')
                return x
            with self.assertRaisesRegex(ValueError,'invalid media'):
                list(prefetched(pool,broken,range(10)))

    def test_job_lock_excludes_duplicate_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'job.lock'
            with job_lock(path) as acquired:
                self.assertTrue(acquired)
                with job_lock(path) as other:self.assertFalse(other)
            with job_lock(path) as after:self.assertTrue(after)

if __name__=='__main__':unittest.main()
