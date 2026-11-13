import sys
from pathlib import Path
import unittest
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_spectre_coefficients import temporal_indices
from p2_avatarforcing_table4_score import fd_definitions,coefficient_correlations,histogram_entropy
from vico_literature_metrics import load_author


class Table4Tests(unittest.TestCase):
    def test_temporal_context_preserves_every_original_frame_once(self):
        for length in [1,2,31,32,33,64,65,1049]:
            recovered=[]
            for start,end,indices in temporal_indices(length,32):
                self.assertEqual(len(indices),end-start+4)
                np.testing.assert_array_equal(indices[2:-2],np.arange(start,end))
                self.assertTrue(np.all((0<=indices)&(indices<length)))
                recovered.extend(indices[2:-2])
            self.assertEqual(recovered,list(range(length)))

    def test_fd_mean_term_variants_have_known_analytic_values(self):
        g=np.zeros((4,2),dtype=np.float64);p=g+np.array([3.,4.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            result=fd_definitions(g,p,load_author())
        self.assertAlmostEqual(result['standard_squared_l2'],25)
        self.assertAlmostEqual(result['paper_mean_norm_unsquared_l2'],5)
        self.assertAlmostEqual(result['paper_mean_norm_l1'],7)

    def test_constant_coefficient_is_not_reported_as_valid_correlation(self):
        a=np.stack([np.arange(20),np.ones(20)],axis=1).astype(np.float32)
        c=coefficient_correlations(a,a)
        self.assertAlmostEqual(c[0],1)
        self.assertTrue(np.isnan(c[1]))

    def test_sid_k15_and_k9_have_distinct_entropy_ceilings(self):
        e=histogram_entropy(np.arange(15),15);p=histogram_entropy(np.arange(9),9)
        self.assertAlmostEqual(e,np.log2(15),delta=1e-4)
        self.assertAlmostEqual(p,np.log2(9),delta=1e-4)
        self.assertGreater(e,p)


if __name__=='__main__':unittest.main()
