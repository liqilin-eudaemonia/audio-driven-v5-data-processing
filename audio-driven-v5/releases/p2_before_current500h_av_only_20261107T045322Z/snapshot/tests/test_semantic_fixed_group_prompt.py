from copy import deepcopy
import unittest
from listen_benchmark import semantic_fixed_group_prompt as fixed
from listen_benchmark import semantic_prompt as base


class FixedGroupTests(unittest.TestCase):
    def record(self):
        return {'sample_id':'case','duration_s':5.,'events':[
            {'event_id':'a','modality':'visual','start':1.,'end':2.,'description':'The listener smiles and nods.'},
            {'event_id':'b','modality':'visual','start':1.2,'end':1.8,'description':'The listener smiles.'}]}
    def plan(self):
        return {'keepers':['a'],'assignments':[
            {'source_id':'a','target_id':'a','quote_index':0,'reason':'Most complete'},
            {'source_id':'b','target_id':'a','quote_index':0,'reason':'Same overlapping smile'}],'conflicts':[]}
    def test_exact_original_quote_and_time(self):
        r=self.record();p=self.plan();answer=fixed.compile_plan(r,p)
        self.assertEqual(answer,base.compile_plan(r,fixed.to_strict_plan(r,p)))
        result=fixed.materialize(r,answer,{'accepted':True,'issues':[]},{})
        self.assertEqual(result['events'][0]['description'],r['events'][0]['description'])
        self.assertEqual((result['events'][0]['start'],result['events'][0]['end']),(1.,2.))
    def test_schema_forces_coverage_and_omission_alarm_eligibility(self):
        schema=fixed.plan_schema(self.record());a=schema['properties']['assignments']
        self.assertEqual(a['minItems'],2);self.assertEqual(a['maxItems'],2)
        self.assertFalse(a['items']);self.assertEqual(a['prefixItems'][0]['properties']['target_id']['enum'],['a'])
        self.assertEqual(set(a['prefixItems'][1]['properties']['target_id']['enum']),{'a','b'})
    def test_missing_source_rejected(self):
        p=self.plan();p['assignments'].pop()
        with self.assertRaises(ValueError):fixed.compile_plan(self.record(),p)
    def test_keeper_cannot_point_to_another_keeper(self):
        p=self.plan();p['keepers'].append('b')
        with self.assertRaisesRegex(ValueError,'target itself'):fixed.compile_plan(self.record(),p)
    def test_support_must_target_explicit_keeper(self):
        p=self.plan();p['assignments'][1]['target_id']='b'
        with self.assertRaisesRegex(ValueError,'not in explicit keepers'):fixed.compile_plan(self.record(),p)
    def test_cannot_hide_nod_or_reorder_assignments(self):
        p=self.plan();p['assignments'][0]['target_id']=None
        with self.assertRaises(ValueError):fixed.compile_plan(self.record(),p)
        p=self.plan();p['assignments'].reverse()
        with self.assertRaisesRegex(ValueError,'reordered'):fixed.compile_plan(self.record(),p)
    def test_different_occurrences_still_rejected(self):
        r=self.record();r['events'][1].update(start=3.,end=4.)
        with self.assertRaisesRegex(ValueError,'does not overlap'):fixed.compile_plan(r,self.plan())


if __name__=='__main__':unittest.main()
