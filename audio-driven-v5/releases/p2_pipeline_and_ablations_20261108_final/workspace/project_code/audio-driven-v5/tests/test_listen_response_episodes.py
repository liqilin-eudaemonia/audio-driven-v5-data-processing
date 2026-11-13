import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.response_episodes import consolidate_responses


def e(i,a,b,text='yeah',anchor=True,mod='audio'):
    value=dict(event_id=i,start=a,end=b,modality=mod,behavior=f'Says "{text}".' if mod=='audio' else 'Nods.',
               categories=['listener_vocalization'] if mod=='audio' else ['head_nod'],
               mapping_status='mapped_candidate',event_unit='word' if anchor else 'source_event',
               source_descriptions=[],review_flags=[])
    if anchor:value['text']=text
    return value


def s(events):return dict(sample_id='s1',duration_s=10,events=events,human_ground_truth=False,training_admitted=False)


class ResponseEpisodeTests(unittest.TestCase):
    def test_reported_four_yeah_observations_become_one_response_not_four_words(self):
        original=s([e('a',4.4,4.6,anchor=False),e('b',4.48,4.56),
                    e('c',4.5,4.7,anchor=False),e('d',4.56,4.96)])
        saved=copy.deepcopy(original);r,t=consolidate_responses(original)
        self.assertEqual(len(r['events']),1)
        episode=r['events'][0]
        self.assertEqual((episode['start'],episode['end']),(4.48,4.96))
        self.assertEqual(episode['member_event_ids'],['a','b','c','d'])
        self.assertEqual(len(episode['word_observations']),2)
        self.assertIsNone(episode['repetition_count'])
        self.assertEqual(episode['behavior'],'Says "yeah".')
        self.assertEqual(original,saved)

    def test_nod_during_continuous_voice_does_not_split_it(self):
        nod=e('nod',4.5,4.7,anchor=False,mod='visual')
        r,_=consolidate_responses(s([e('a',4.4,4.6),e('b',4.6,4.96),nod]))
        self.assertEqual(len(r['events']),2)
        audio=next(e for e in r['events'] if e['modality']=='audio')
        self.assertEqual(audio['overlapping_visual_event_ids'],['nod'])
        self.assertEqual(next(e for e in r['events'] if e['modality']=='visual'),nod)

    def test_separate_okays_stay_separate_and_broad_report_cannot_bridge_gap(self):
        rows=[e('a',1.78,2.10,'Okay'),e('b',4.08,4.40,'Okay'),e('wide',1.5,4.5,'okay',False)]
        r,_=consolidate_responses(s(rows))
        self.assertEqual([(e['start'],e['end']) for e in r['events']],[(1.78,2.10),(4.08,4.40)])
        self.assertEqual(len(r['unresolved_observations']),1)
        self.assertEqual(r['unresolved_observations'][0]['observation']['event_id'],'wide')

    def test_different_content_is_not_merged_even_when_overlapping(self):
        r,_=consolidate_responses(s([e('a',1,2,'yeah'),e('b',1.5,2.5,'okay')]))
        self.assertEqual(len(r['events']),2)

    def test_model_interval_cannot_grow_anchor_group_to_absorb_next_response(self):
        r,_=consolidate_responses(s([e('a',1,1.3),e('b',2,2.3),
            e('m1',1.2,1.9,anchor=False),e('m2',1.8,2.1,anchor=False)]))
        self.assertEqual([(e['start'],e['end']) for e in r['events']],[(1,1.3),(2,2.3)])

    def test_unanchored_overlap_merges_but_gap_does_not(self):
        r,t=consolidate_responses(s([e('a',1,1.4,anchor=False),e('b',1.2,1.6,anchor=False),
                                    e('c',1.8,2,anchor=False)]))
        self.assertEqual(len(r['events']),2)
        self.assertEqual((r['events'][0]['start'],r['events'][0]['end']),(1,1.6))
        self.assertEqual(sum(len(v['observations']) for v in t.values()),3)

    def test_uncertain_and_out_of_scope_language_remains_untouched(self):
        a=e('a',1,2);a.update(mapping_status='needs_review',categories=[])
        r,t=consolidate_responses(s([a]))
        self.assertEqual(r['events'],[a]);self.assertFalse(t)


if __name__=='__main__':unittest.main()
