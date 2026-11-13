import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.integration import integrate

class IntegrationTests(unittest.TestCase):
    def test_word_offsets_repetitions_and_invalid_intervals(self):
        windows=[dict(sample_id='w0',window={'start_s':0,'end_s':3},words=[dict(text='okay',start_time=1.76,end_time=2.08)]),
                 dict(sample_id='w1',window={'start_s':1,'end_s':4},words=[dict(text='Okay',start_time=.8,end_time=1.12)]),
                 dict(sample_id='w4',window={'start_s':4,'end_s':7},words=[dict(text='Okay',start_time=.08,end_time=.4),dict(text='Okay',start_time=.4,end_time=.4)])]
        result=integrate([],[],windows,9)
        self.assertEqual(len(result['events']),2)
        self.assertAlmostEqual(result['events'][0]['start'],1.78)
        self.assertAlmostEqual(result['events'][0]['end'],2.1)
        self.assertEqual(result['events'][0]['support'],2)
        self.assertEqual(result['events'][1]['start'],4.08)
        self.assertEqual(len(result['invalid_words']),1)

    def test_cross_modal_overlap_preserves_independent_boundaries(self):
        visual=[dict(id='w',window={'start_s':3,'end_s':6},events=[dict(start=4.2,end=4.85,modality='visual',behavior='nod',valid=True)])]
        words=[dict(sample_id='a',window={'start_s':4,'end_s':7},words=[dict(text='yeah',start_time=.4,end_time=.68)])]
        result=integrate(visual,[],words,9)
        self.assertEqual(len(result['events']),2)
        self.assertEqual(result['events'][0]['start'],4.2)
        self.assertEqual(result['events'][0]['end'],4.85)
        self.assertEqual(result['events'][1]['start'],4.4)
        self.assertEqual(len(result['links']),1)
        self.assertFalse(result['human_ground_truth'])

if __name__=='__main__':unittest.main()
