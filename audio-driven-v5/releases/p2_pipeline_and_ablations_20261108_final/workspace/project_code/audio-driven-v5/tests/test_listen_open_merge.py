import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.open_merge import consolidate, atomic_consolidate

def c(i,start,end,text='Smiling',source='qwen3_vl'):
    return dict(id=i,start=start,end=end,behavior=text,source=source,modality='audio' if source!='qwen3_vl' else 'visual',support=1)
def proposal(groups,excluded=None):return dict(finish_reason='stop',parsed=dict(groups=groups,excluded=excluded or []))
def group(ids,text='Smiles',mode='same_occurrence'):return dict(members=ids,behavior=text,mode=mode,uncertain=False)

class OpenMergeTests(unittest.TestCase):
    def test_new_open_predicate_is_not_forced_into_existing_vocabulary(self):
        norm={'Twirls a pen':{'id':'d1','value':{'atoms':[{'signature':'twirl pen','behavior':'Twirls a pen.','temporal_kind':'occurrence','uncertain':False}],'excluded_reason':''}}}
        r=atomic_consolidate([c('a',0,1,'Twirls a pen')],norm,2,rules={'signature_aliases':{'smiling':'smile'}})
        self.assertEqual(r['events'][0]['behavior'],'Twirls a pen.')

    def test_opposite_directions_and_different_words_do_not_merge(self):
        for source,texts,signature in [('qwen3_vl',['Head tilts left','Head tilts right'],'tilt head'),('omni_audio',["Says 'okay'","Says 'yeah'"],'vocalize')]:
            norm={t:{'id':t,'value':{'atoms':[{'signature':signature,'behavior':t,'temporal_kind':'occurrence','uncertain':False}],'excluded_reason':''}} for t in texts}
            r=atomic_consolidate([c('a',0,1,texts[0],source),c('b',.05,1.05,texts[1],source)],norm,2)
            self.assertEqual(len(r['events']),2)

    def test_continuous_microgesture_process_retains_internal_transitions(self):
        texts=['Opens mouth','Closes mouth'];norm={t:{'id':t,'value':{'atoms':[{'signature':k,'behavior':t,'temporal_kind':'occurrence','uncertain':False}],'excluded_reason':''}} for t,k in zip(texts,['open mouth','close mouth'])}
        rules={'compound_process_predicates':['open mouth','close mouth'],'compound_process_description':'Repeated mouth movements.'}
        r=atomic_consolidate([c('a',0,.2,texts[0]),c('b',.2,.4,texts[1]),c('c',.4,.6,texts[0])],norm,1,rules=rules)
        self.assertEqual(len(r['events']),1)
        trace=r['trace'][r['events'][0]['event_id']]
        self.assertEqual([x['start'] for x in trace['atoms']],[0,.2,.4])

    def test_atomic_composite_splits_actions_without_losing_source(self):
        norm={'Smiles and nods':{'id':'d1','value':{'atoms':[
            {'signature':'smile','behavior':'Smiles.','temporal_kind':'state','uncertain':False},
            {'signature':'nod','behavior':'Nods.','temporal_kind':'occurrence','uncertain':False}], 'excluded_reason':''}}}
        r=atomic_consolidate([c('a',.5,1.5,'Smiles and nods')],norm,2)
        self.assertEqual(len(r['events']),2)
        self.assertTrue(all(t['source_candidate_ids']==['a'] for t in r['trace'].values()))

    def test_atomic_states_merge_but_repeated_occurrences_remain(self):
        for kind,expected in [('state',1),('occurrence',2)]:
            norm={'Smiling':{'id':'d1','value':{'atoms':[{'signature':'unrestricted free predicate','behavior':'Observable action.','temporal_kind':kind,'uncertain':False}],'excluded_reason':''}}}
            r=atomic_consolidate([c('a',0,1),c('b',1,2)],norm,2)
            self.assertEqual(len(r['events']),expected)

    def test_atomic_asr_preserves_two_words_and_only_attaches_matching_atom(self):
        norm={'Says okay and laughs':{'id':'d1','value':{'atoms':[
            {'signature':'say okay','behavior':'Says "Okay".','temporal_kind':'occurrence','uncertain':False},
            {'signature':'laugh','behavior':'Laughs.','temporal_kind':'occurrence','uncertain':False}],'excluded_reason':''}}}
        r=atomic_consolidate([c('a',1,1.3,'Okay','asr_aligner'),c('b',2,2.3,'Okay','asr_aligner'),c('m',.8,1.8,'Says okay and laughs','omni_audio')],norm,3)
        self.assertEqual(len(r['events']),3)
        words=[e for e in r['events'] if 'text' in e]
        self.assertEqual([e['start'] for e in words],[1,2])
        self.assertEqual(len(r['trace'][words[0]['event_id']]['atom_ids']),2)

    def test_atomic_exclusions_and_missing_normalizations_account_for_all_inputs(self):
        norm={'Neutral posture':{'id':'d1','value':{'atoms':[],'excluded_reason':'No observable change'}}}
        r=atomic_consolidate([c('a',0,1,'Neutral posture'),c('b',1,2,'Unknown observed action')],norm,2)
        self.assertEqual(len(r['events']),1);self.assertEqual(r['excluded'][0]['source_candidate_id'],'a')
        self.assertIn('normalization_missing_or_invalid_original_retained',r['events'][0]['review_flags'])

    def test_continuous_paraphrases_join_and_preserve_provenance(self):
        r=consolidate([c('a',.5,1.5),c('b',1.4,2.5,'Mouth corners raised')],{'visual':proposal([group(['a','b'],mode='continuous_state')])},3)
        self.assertEqual(len(r['events']),1);self.assertEqual((r['events'][0]['start'],r['events'][0]['end']),(.5,2.5))
        self.assertEqual(r['events'][0]['source_candidate_ids'],['a','b'])
    def test_repeated_words_cannot_be_merged_or_discarded(self):
        candidates=[c('a',1,1.3,'Okay','asr_aligner'),c('b',2,2.3,'Okay','asr_aligner')]
        for p in [proposal([group(['a','b'],'Says Okay')]),proposal([],excluded=[{'members':['a','b'],'reason':'redundant'}])]:
            r=consolidate(candidates,{'audio':p},3);self.assertEqual(len(r['events']),2)
            self.assertEqual([e['start'] for e in r['events']],[1,2])
    def test_asr_anchor_keeps_boundary_when_other_model_disagrees(self):
        r=consolidate([c('a',1,1.3,'Okay','asr_aligner'),c('b',.8,1.8,'Says okay','omni_audio')],{'audio':proposal([group(['a','b'],'Says "Okay".')])},3)
        self.assertEqual(len(r['events']),1);self.assertEqual((r['events'][0]['start'],r['events'][0]['end']),(1,1.3))
        self.assertIn('source_boundary_disagreement',r['events'][0]['review_flags'])
    def test_time_gap_prevents_silent_collapse(self):
        for mode in ['same_occurrence','continuous_state']:
            r=consolidate([c('a',0,1),c('b',2,3)],{'visual':proposal([group(['a','b'],mode=mode)])},3)
            self.assertEqual(len(r['events']),2)
    def test_conflicting_duplicate_boundaries_choose_observed_interval(self):
        r=consolidate([c('a',0,1),c('b',.4,1.7)],{'visual':proposal([group(['a','b'])])},2)
        self.assertIn((r['events'][0]['start'],r['events'][0]['end']),[(0,1),(.4,1.7)])
        self.assertIn('source_boundary_disagreement',r['events'][0]['review_flags'])
    def test_duplicate_unknown_or_missing_ids_cannot_lose_inputs(self):
        r=consolidate([c('a',0,1),c('b',.4,1.7)],{'visual':proposal([group(['a']),group(['a']),group(['invented'])])},2)
        self.assertEqual(len(r['events']),2);self.assertEqual({x for e in r['events'] for x in e['source_candidate_ids']},{'a','b'})
        self.assertFalse(r['training_admitted'])
    def test_cross_modality_ids_cannot_merge(self):
        r=consolidate([c('a',0,1),c('b',.4,.8,'yeah','asr_aligner')],{'visual':proposal([group(['a','b'])])},2)
        self.assertEqual(len(r['events']),2)

if __name__=='__main__':unittest.main()
