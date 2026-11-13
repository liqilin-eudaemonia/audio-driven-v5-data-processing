"""Resource exclusion and completed-seed aggregation contracts for this run."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_seed_eval',ROOT/'scripts/p2_seamless200_pipeline_two_seeds_20261112.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PairedSeedContracts(unittest.TestCase):
    def cluster(self):
        return {'nodes':[dict(name=name,host=host,gpus=list(range(count))) for name,host,count in
            [('node12','172.29.252.37',8),('node8','172.29.104.237',8),
             ('node4','172.29.225.51',8),('node11','172.29.189.119',4)]]}
    def test_requested_28_cards(self):
        module.validate_cluster(self.cluster())
    def test_reserved_gpu_is_rejected(self):
        for gpu in range(4,8):
            c=self.cluster();c['nodes'][3]['gpus'].append(gpu)
            with self.assertRaises(AssertionError):module.validate_cluster(c)
    def test_node_permutation_is_rejected(self):
        c=self.cluster();c['nodes'][0],c['nodes'][3]=c['nodes'][3],c['nodes'][0]
        with self.assertRaises(AssertionError):module.validate_cluster(c)
    def test_mean_of_completed_global_scores(self):
        r={137:{'metrics':{'fid':10.,'fvd':100.,'psnr':20.}},238:{'metrics':{'fid':30.,'fvd':200.,'psnr':22.}}}
        self.assertEqual(module.average_metrics(r),{'fid':20.,'fvd':150.,'psnr':21.})
    def test_missing_seed_is_not_a_finished_average(self):
        with self.assertRaises(AssertionError):module.average_metrics({137:{'metrics':{'fid':10.}}})
    def test_unavailable_audio_score_is_not_imputed(self):
        r={137:{'metrics':{'utmos':None,'fid':10.}},238:{'metrics':{'utmos':3.,'fid':12.}}}
        self.assertIsNone(module.average_metrics(r)['utmos'])
        self.assertEqual(module.average_metrics(r)['fid'],11.)
    def test_nonfinite_score_is_unavailable(self):
        r={137:{'metrics':{'utmos':float('nan')}},238:{'metrics':{'utmos':3.}}}
        self.assertIsNone(module.average_metrics(r)['utmos'])

if __name__=='__main__':unittest.main()
