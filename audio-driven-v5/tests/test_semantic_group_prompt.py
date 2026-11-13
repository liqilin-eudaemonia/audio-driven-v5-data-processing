from copy import deepcopy
import unittest
from listen_benchmark import semantic_group_prompt as group
from listen_benchmark import semantic_prompt as base


class GroupPlanTests(unittest.TestCase):
    def record(self):
        return {'sample_id':'case','duration_s':5.,'events':[
            {'event_id':'a','modality':'visual','start':1.,'end':2.,'description':'The listener smiles and nods.'},
            {'event_id':'b','modality':'visual','start':1.2,'end':1.8,'description':'The listener smiles.'},
            {'event_id':'audio','modality':'audio','start':3.,'end':3.3,'description':'yeah','text':'yeah'}]}
    def plan(self):
        return {'groups':[{'keep_id':'a','members':[{'source_id':'a','quote_index':0},{'source_id':'b','quote_index':0}],
                           'reason':'Same overlapping smile; retain nod.'}],'inactive':[],'conflicts':[]}
    def test_group_translation_matches_strict_plan_and_preserves_audio(self):
        r=self.record();p=self.plan();before=deepcopy(r)
        ans=group.compile_plan(r,p)
        self.assertEqual(ans,base.compile_plan(r,group.to_strict_plan(r,p)))
        result=group.materialize(r,ans,{'accepted':True,'issues':[]},{})
        self.assertEqual(result['events'][-1],r['events'][-1]);self.assertEqual(result['events'][0]['description'],r['events'][0]['description'])
        self.assertEqual(r,before)
    def test_no_chains_or_repeated_members(self):
        p=self.plan();p['groups'].append({'keep_id':'b','members':[{'source_id':'b','quote_index':0}],'reason':'duplicate'})
        with self.assertRaisesRegex(ValueError,'multiple'):group.compile_plan(self.record(),p)
    def test_representative_must_belong_to_group(self):
        p=self.plan();p['groups'][0]['members']=p['groups'][0]['members'][1:]
        with self.assertRaisesRegex(ValueError,'own group'):group.compile_plan(self.record(),p)
    def test_no_invented_quote_or_omitted_source(self):
        for index in [-1,20,True]:
            p=self.plan();p['groups'][0]['members'][0]['quote_index']=index
            with self.assertRaises(ValueError):group.compile_plan(self.record(),p)
        p=self.plan();p['groups'][0]['members']=p['groups'][0]['members'][:1]
        with self.assertRaisesRegex(ValueError,'omitted'):group.compile_plan(self.record(),p)
    def test_nod_cannot_be_hidden_in_smile(self):
        p=self.plan();p['groups'][0]['keep_id']='b'
        with self.assertRaisesRegex(ValueError,'physical action'):group.compile_plan(self.record(),p)
    def test_separate_repeated_occurrences_cannot_be_grouped(self):
        r=self.record();r['events'][1].update(start=3.,end=4.)
        with self.assertRaisesRegex(ValueError,'does not overlap'):group.compile_plan(r,self.plan())
    def test_review_sees_actual_intervals_without_modifying_proposal(self):
        r=self.record();ans=group.compile_plan(r,self.plan());old=deepcopy(ans)
        v=group.review_payload(r,ans)
        self.assertEqual(v['proposal']['visual_events'][0]['intervals'],[{'start':1.,'end':2.}]);self.assertEqual(ans,old)
        self.assertEqual(len(v['source_coverage']),2)
    def test_schema_preflight_still_rejects_inference_between_actions(self):
        r=self.record();r['events'][0]['description']='The listener smiles, suggesting enjoyment. Then he nods.'
        with self.assertRaises(ValueError):group.plan_schema(r)


if __name__=='__main__':unittest.main()
