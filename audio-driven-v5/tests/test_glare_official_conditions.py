import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import generation_rows,read,IntegrityError
from prepare_glare_official_conditions import OfficialPolicy


class OfficialConditionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=OfficialPolicy(read(ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/protocol.json')['extractor'])
    def record(self,raw):
        gen=generation_rows([[8,99]],[99],64)[0];gen.update(eos_ids=[99],max_new_tokens=64)
        return dict(analysis=dict(status='diagnostic_only',raw_text=raw,accepted=False),generation=gen,
                    official_normalized_text=self.policy.normalize(raw,max_time=2.4))
    def test_official_null_keeps_pair_even_when_raw_is_invalid(self):
        for raw in ['NULL','Sorry, cannot analyze this audio.',r'0.00, 0.50, 0.80\n0.60, 1.00, 0.90','2.34, 2.79, 0.50']:
            result=self.policy.map_pair(dict(start=0.,end=2.4,duration=2.4),{0:self.record(raw)})
            self.assertEqual(result['decision'],'keep');self.assertFalse(result['intensity'].any())
    def test_native_normalizer_and_frame_encoder_are_used_without_strict_filter(self):
        for raw in ['junk\n0.00, 1.00, 0.80\nNULL','0.00, 1.00, 0.80\n0.50, 1.50, 1.50','1.39, 1.70, 0.50']:
            result=self.policy.map_pair(dict(start=0.,end=2.4,duration=2.4),{0:self.record(raw)})
            np.testing.assert_array_equal(result['intensity'],self.policy.vector(self.policy.normalize(raw,2.4),60))
            self.assertEqual(result['decision'],'keep')
    def test_skipped_tail_zero_but_missing_and_execution_errors_are_distinct(self):
        pair=dict(start=2.4,end=3.4,duration=1.)
        tail=dict(analysis=dict(status='uncovered_source_tail'))
        self.assertEqual(self.policy.map_pair(pair,{})['decision'],'pending')
        result=self.policy.map_pair(pair,{1:tail});self.assertEqual(result['decision'],'keep');self.assertFalse(result['intensity'].any())
        self.assertEqual(self.policy.map_pair(pair,{1:dict(analysis=dict(status='input_error_exhausted'))})['decision'],'failed_execution')
        early=self.record('Sorry');early['generation']['early_reject']=True
        with self.assertRaises(IntegrityError):self.policy.map_pair(pair,{1:early})


if __name__=='__main__':unittest.main()
