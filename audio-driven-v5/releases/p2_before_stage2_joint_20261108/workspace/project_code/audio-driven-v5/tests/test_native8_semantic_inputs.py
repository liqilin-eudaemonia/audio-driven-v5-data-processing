from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_native8_nod_semantic_pilot import prepare


class SemanticInputTests(unittest.TestCase):
    def bundle(self):
        event={'start':.5,'end':1.,'behavior':'Head nodding.'}
        return {'sample_id':'sample','duration_s':3.,'vlm_observations':[
            {'view':'full','start':0.,'status':'valid','parsed':{'events':[event]}},
            {'view':'w000','start':0.,'status':'valid','parsed':{'events':[deepcopy(event)]}}],
            'nod_candidates':[{'event_id':'nod-1','start':.5,'end':1.,'description':'Head nodding.'}]}

    def test_raw_and_light_preserve_every_observation_and_detector_source(self):
        b=self.bundle();before=deepcopy(b)
        raw=prepare(b,deduplicate_exact=False);light=prepare(b)
        self.assertEqual(len(raw['events']),3);self.assertEqual(len(light['events']),2)
        self.assertEqual(sum(len(e['original_observations']) for e in light['events']),3)
        self.assertEqual({e['source_type'] for e in light['events']},{'vlm_video8','glare_nod_recall'})
        self.assertEqual(b,before)

    def test_separate_occurrences_are_not_merged(self):
        b=self.bundle();b['vlm_observations'][1]['start']=1.
        self.assertEqual(len(prepare(b)['events']),3)

    def test_unsafe_extraction_is_retained_for_runner_preflight(self):
        b=self.bundle();b['vlm_observations'][0]['parsed']['events'][0]['behavior']='He smiles, suggesting enjoyment. Then he nods.'
        with self.assertRaises(ValueError):prepare(b)
        result=prepare(b,validate_plan=False)
        self.assertIn('Then he nods.',result['events'][0]['description'])


if __name__=='__main__':unittest.main()
