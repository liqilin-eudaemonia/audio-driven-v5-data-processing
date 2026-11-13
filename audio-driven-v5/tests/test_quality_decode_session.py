import unittest
from types import SimpleNamespace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.quality_decode_session import Codec,Container,Stream


class DecodeSessionTests(unittest.TestCase):
    def test_identical_thread_setting_is_safe_after_open(self):
        class C:
            is_open=True
            @property
            def thread_count(self):return 1
            @thread_count.setter
            def thread_count(self,v):raise RuntimeError('opened')
        c=Codec(C());c.thread_count=1
        with self.assertRaises(RuntimeError):c.thread_count=2

    def container(self,last_t):
        stream=Stream(SimpleNamespace(time_base=1,codec_context=SimpleNamespace()))
        calls=[]
        inner=SimpleNamespace(seek=lambda *a,**k:calls.append((a,k)),decode=lambda s:iter([]))
        c=Container.__new__(Container);c.inner=inner;c.iterator=iter([])
        c.last=SimpleNamespace(pts=last_t,time_base=1);c.replay=False
        c.stats={'seeks':0,'seeks_avoided':0}
        return c,stream,calls

    def test_adjacent_forward_interval_replays_first_boundary_frame(self):
        c,s,calls=self.container(5)
        c.seek(5,stream=s,backward=True,any_frame=False)
        self.assertEqual(calls,[])
        self.assertEqual([f.pts for f in c.decode(s)],[5])
        self.assertEqual(c.stats['seeks_avoided'],1)

    def test_backward_and_large_gap_use_original_seek(self):
        for target in (4,8):
            c,s,calls=self.container(5)
            c.seek(target,stream=s,backward=True,any_frame=False)
            self.assertEqual(calls[0][0],(target,))
            self.assertIs(calls[0][1]['stream'],s.inner)
            self.assertEqual(c.stats['seeks'],1)


if __name__=='__main__':unittest.main()
