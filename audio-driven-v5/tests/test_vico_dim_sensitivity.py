import importlib.util
from pathlib import Path
import unittest

import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/p2_vico_dim_sensitivity.py'
spec=importlib.util.spec_from_file_location('vico_dim_sensitivity',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SensitivityTests(unittest.TestCase):
    def features(self):
        a=np.arange(5*56,dtype=np.float32).reshape(5,56)
        return {('a',r):a.copy() for r in ['gt','speaker','m1','m2']}

    def test_offset_changes_speaker_only_and_never_adds_listener_t0(self):
        f=self.features()
        g,x,p,c=module.build_case(f,['a'],['m1','m2'],1)
        g0,x0,p0,c0=module.build_case(f,['a'],['m1','m2'],0)
        np.testing.assert_array_equal(g[0],g0[0]);np.testing.assert_array_equal(p['m1'][0],p0['m1'][0])
        np.testing.assert_array_equal(x[0],f['a','speaker'][:-1]);np.testing.assert_array_equal(x0[0],f['a','speaker'][1:])
        self.assertEqual(c[0]['listener_frames'],[1,2,3,4]);self.assertEqual(c0[0]['speaker_frames'],[1,2,3,4])

    def test_missing_face_uses_common_mask_for_both_methods(self):
        f=self.features();f['a','m1'][2]=np.nan
        g,x,p,c=module.build_case(f,['a'],['m1','m2'],1)
        self.assertEqual(c[0]['listener_frames'],[1,3,4]);self.assertEqual(c[0]['excluded_frames'],1)
        self.assertEqual(len(g[0]),len(p['m2'][0]));self.assertTrue(np.isfinite(p['m1'][0]).all())

    def test_loader_projection_keeps_explicit_repeated_rows(self):
        g,x,p,c=module.build_case(self.features(),['a','a'],['m1','m2'],1)
        self.assertEqual(len(g),2);self.assertEqual([r['id'] for r in c],['a','a'])

    def test_coefficient_mean_offset_affects_flatten_but_not_temporal_variance(self):
        a=np.array([[0.,0.],[2.,2.]],dtype=np.float32)
        before=module.variance_statistics([a]);after=module.variance_statistics([a+np.array([0.,10.])])
        self.assertEqual(before['pooled_per_coefficient_mean'],after['pooled_per_coefficient_mean'])
        self.assertNotEqual(before['pooled_flatten'],after['pooled_flatten'])

    def test_equal_clip_and_global_entropy_measure_different_things(self):
        self.assertAlmostEqual(module.entropy(np.array([0,0,1,1]),2),1.,places=5)
        self.assertAlmostEqual(np.mean([module.entropy(np.array([0,0]),2),module.entropy(np.array([1,1]),2)]),0.,places=5)
        with self.assertRaises(ValueError):module.entropy(np.array([],dtype=int),2)

    def test_invalid_streams_and_offsets_fail_closed(self):
        f=self.features();f['a','gt']=f['a','gt'][:-1]
        with self.assertRaises(ValueError):module.build_case(f,['a'],['m1','m2'],1)
        with self.assertRaises(ValueError):module.build_case(self.features(),['a'],['m1','m2'],2)


if __name__=='__main__':unittest.main()
