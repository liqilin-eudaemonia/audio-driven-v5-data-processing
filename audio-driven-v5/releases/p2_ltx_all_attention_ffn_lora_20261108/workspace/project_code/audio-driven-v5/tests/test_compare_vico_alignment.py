"""Source timing regression for the independently adapted L2L inference loop."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'compare/deployment/vico_l2l.py'
spec=importlib.util.spec_from_file_location('l2l_adapter',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AlignmentTests(unittest.TestCase):
    def test_source_time_is_preserved_after_every_window_shift(self):
        # Distinct time sentinels expose the official t-vs4t slicing bug.
        frames=np.arange(112,dtype=np.float32)
        speaker=np.broadcast_to(frames[None,:,None],(3,112,56)).copy()
        audio=np.broadcast_to(np.repeat(frames,4)[None,:,None],(3,448,128)).copy()
        for start in range(0,73,8):
            x,a=m.aligned_window(speaker,audio,start)
            np.testing.assert_array_equal(x[:,:,0],a[:,::4,0])
            self.assertEqual(x[0,0,0],start)
            self.assertEqual(x.shape,(3,40,56));self.assertEqual(a.shape,(3,160,128))

    def test_incomplete_future_context_is_rejected(self):
        with self.assertRaises(ValueError):m.aligned_window(np.zeros((1,39,56)),np.zeros((1,156,128)),0)


if __name__=='__main__':unittest.main()
