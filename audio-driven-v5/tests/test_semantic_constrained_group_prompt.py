import unittest
from listen_benchmark import semantic_fixed_group_prompt as fixed
from listen_benchmark import semantic_constrained_group_prompt as constrained


class ConstraintExposureTests(unittest.TestCase):
    def record(self):
        return {'sample_id':'case','duration_s':5.,'events':[
            {'event_id':'nod','modality':'visual','start':1.,'end':2.,'description':'The listener smiles and nods.'},
            {'event_id':'smile','modality':'visual','start':1.2,'end':1.8,'description':'The listener smiles.'}]}
    def test_existing_mandatory_keeper_is_explicit(self):
        r=self.record();p=constrained.source_payload(r);s=constrained.plan_schema(r)
        self.assertEqual(p['mandatory_keeper_ids'],['nod'])
        self.assertEqual(p['visual_sources'][0]['allowed_representatives'],['nod'])
        self.assertEqual(s['properties']['keepers']['prefixItems'],[{'type':'string','const':'nod'}])
        self.assertEqual(s['properties']['keepers']['items']['enum'],['smile'])
    def test_assignment_constraints_remain_identical(self):
        r=self.record()
        self.assertEqual(fixed.plan_schema(r)['properties']['assignments'],constrained.plan_schema(r)['properties']['assignments'])
    def test_optional_cluster_is_visible_as_uncovered(self):
        r=self.record();r['events'][0]['description']='The listener smiles.'
        p=constrained.source_payload(r)
        self.assertEqual(p['mandatory_keeper_ids'],[]);self.assertEqual(p['uncovered_by_mandatory'],['nod','smile'])
    def test_unsafe_quote_remains_a_per_sample_preflight_failure(self):
        r=self.record();r['events'][0]['description']='The listener smiles, suggesting enjoyment. Then he nods.'
        p=constrained.source_payload(r)
        self.assertEqual(p['visual_sources'][0]['allowed_representatives'],[])
        with self.assertRaises(ValueError):constrained.plan_schema(r)


if __name__=='__main__':unittest.main()
