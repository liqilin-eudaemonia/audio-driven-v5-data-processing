import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/p2_vico_dim_best10.py'
spec=importlib.util.spec_from_file_location('vico_dim_best10',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Best10Tests(unittest.TestCase):
    def test_full56_choice_is_one_candidate_not_separate_exp_pose_winners(self):
        from vico_literature_metrics import load_author
        gt=np.tile(np.arange(56,dtype=np.float32)*.1,(4,1))
        a=gt.copy();a[:,:6]+=10
        b=gt.copy();b[:,6:]+=.5
        index,distances=module.choose_candidate(gt,[a,b],load_author())
        self.assertEqual(index,1)
        self.assertLess(distances[1],distances[0])
        self.assertTrue(np.array_equal(b[:,:6],gt[:,:6]))
        self.assertFalse(np.array_equal(b[:,6:],gt[:,6:]))

    def test_equal_distances_keep_first_candidate_like_author_loop(self):
        from vico_literature_metrics import load_author
        gt=np.tile(np.arange(56,dtype=np.float32)*.1,(4,1))
        index,values=module.choose_candidate(gt,[gt.copy(),gt.copy()],load_author())
        self.assertEqual(index,0);self.assertEqual(values[0],values[1])

    def test_missing_or_misaligned_candidates_are_not_silently_dropped(self):
        from vico_literature_metrics import load_author
        gt=np.zeros((4,56),dtype=np.float32);author=load_author()
        with self.assertRaises(ValueError):module.choose_candidate(gt,[],author)
        with self.assertRaises(ValueError):module.choose_candidate(gt,[gt[:-1]],author)
        broken=gt.copy();broken[1,0]=np.nan
        with self.assertRaises(ValueError):module.choose_candidate(gt,[broken],author)

    def test_only_authorized_gpu_uuid_and_capacity_are_accepted(self):
        with patch.object(module.subprocess,'check_output',return_value=module.UUID+', 45022'):
            self.assertEqual(module.gpu_capacity(88000),(45022,False))
        with patch.object(module.subprocess,'check_output',return_value=module.UUID+', 88000'):
            self.assertEqual(module.gpu_capacity(88000),(88000,True))
        with patch.object(module.subprocess,'check_output',return_value='wrong-card, 140000'):
            with self.assertRaises(ValueError):module.gpu_capacity(88000)


if __name__=='__main__':unittest.main()
