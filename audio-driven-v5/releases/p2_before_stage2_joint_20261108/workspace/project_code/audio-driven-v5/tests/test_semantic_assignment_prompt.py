from copy import deepcopy
import unittest
from listen_benchmark import semantic_assignment_prompt as module
from listen_benchmark import semantic_fixed_group_prompt as fixed


class AssignmentTests(unittest.TestCase):
    def record(self):
        return {'sample_id':'case','duration_s':5.,'events':[
            {'event_id':'a','modality':'visual','start':1.,'end':2.,'description':'The listener smiles and nods.'},
            {'event_id':'b','modality':'visual','start':1.2,'end':1.8,'description':'The listener smiles.'}]}

    def proposal(self):
        return {'assignments':[
            {'source_id':'a','target_id':'a','quote_index':0,'reason':'Most complete'},
            {'source_id':'b','target_id':'a','quote_index':0,'reason':'Same smile'}], 'conflicts':[]}

    def test_equivalent_to_explicit_keeper_plan_without_mutation(self):
        r=self.record(); p=self.proposal(); original=deepcopy(p)
        self.assertEqual(module.compile_plan(r,p),fixed.compile_plan(r,dict(p,keepers=['a'])))
        self.assertEqual(p,original)

    def test_self_keep_needs_no_second_declaration(self):
        p=self.proposal(); p['assignments'][1]['target_id']='b'
        answer=module.compile_plan(self.record(),p)
        result=module.materialize(self.record(),answer,{'accepted':True,'issues':[]},{})
        self.assertEqual(len(result['events']),2)

    def test_cycle_and_unknown_target_rejected_without_repair(self):
        for target in ['b','missing']:
            p=self.proposal();p['assignments'][0]['target_id']=target
            with self.assertRaises(ValueError):module.compile_plan(self.record(),p)

    def test_omission_and_disjoint_occurrences_still_rejected(self):
        p=self.proposal();p['assignments'].pop()
        with self.assertRaises(ValueError):module.compile_plan(self.record(),p)
        r=self.record();r['events'][1].update(start=3.,end=4.)
        with self.assertRaises(ValueError):module.compile_plan(r,self.proposal())
        p=self.proposal();p['assignments'][0]['target_id']='b';p['assignments'][1]['target_id']='b'
        with self.assertRaises(ValueError):module.compile_plan(self.record(),p)

    def test_schema_preserves_every_assignment_restriction(self):
        r=self.record();s=module.plan_schema(r)
        self.assertNotIn('keepers',s['properties'])
        self.assertNotIn('keepers',s['required'])
        self.assertEqual(s['properties']['assignments'],fixed.plan_schema(r)['properties']['assignments'])


if __name__=='__main__':unittest.main()
