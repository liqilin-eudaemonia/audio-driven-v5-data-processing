import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import rolling_video as video

def context(tag,index,latents=0,published=(),scored=False):
    return dict(tag=tag,index=index,done=False,finalizing=False,smoke=True,latents=latents,
        binder=SimpleNamespace(published=set(published),committed=set(),smoke=lambda:True),scored=lambda r:scored)

class RollingChecks(unittest.TestCase):
    def test_fast_gpu_takes_other_group_while_slow_gpu_is_busy(self):
        slow=context('slow',0,64);next_group=context('next',1)
        slow['binder'].committed=set(range(64))
        active=[dict(key=('slow','render',0),physical=0,stage='render',context=slow)]
        selected=video.choose([slow,next_group],active,1)
        self.assertIs(selected[0],next_group);self.assertEqual(selected[1],'render')
        self.assertIsNone(video.choose([slow,next_group],active,0))

    def test_ready_metric_does_not_wait_for_other_groups(self):
        slow=context('slow',0,64);ready=context('ready',1,64,[5])
        active=[dict(key=('slow','render',0),physical=0,stage='render',context=slow)]
        selected=video.choose([slow,ready],active,2)
        self.assertIs(selected[0],ready);self.assertEqual(selected[1:4],('metric',5,False))

    def test_exclusive_gpu_lease_is_shared_by_dispatchers(self):
        with tempfile.TemporaryDirectory()as folder,patch.object(video,'WORK',Path(folder)):
            first=video.acquire(7);self.assertIsNotNone(first)
            self.assertIsNone(video.acquire(7));other=video.acquire(8);self.assertIsNotNone(other);other.close()
            first.close();again=video.acquire(7);self.assertIsNotNone(again);again.close()

    def test_decode_and_metric_concurrency_limits(self):
        c=context('a',0,64,[0]);c['binder'].committed=set(range(60))
        active=[dict(key=('a','decode',0,i),physical=0,stage='decode',context=c)for i in range(2)]
        self.assertIsNone(video.choose([c],active,0,decode_limit=2))

if __name__=='__main__':unittest.main()
