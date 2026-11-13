import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.integration_pilot import integrate

def obs(i,a,b,text='yeah',modality='audio',word=True):
 return dict(event_id=i,start=a,end=b,text=text if word else None,behavior='Says "'+text+'".' if word else text,
             modality=modality,categories=['listener_vocalization'] if modality=='audio' else ['head_nod'],
             mapping_status='mapped_candidate',event_unit='word' if word else 'source_event',review_flags=[])
def decisions(rows,windows):
 return [dict(open_event_id=e['event_id'],origin={'signature':e['behavior']},source_candidates=[dict(id=e['event_id'],sources=[dict(source='asr' if e['event_unit']=='word' else 'model',window_id=w)])]) for e,w in zip(rows,windows)]
def run(rows,windows):return integrate(dict(sample_id='s',duration_s=10,events=rows),decisions(rows,windows),[])

class IntegrationTests(unittest.TestCase):
 def test_cross_window_duplicate_reduces_without_union(self):
  r=run([obs('a',1,1.5),obs('b',1.1,1.6)],['w0','w1']);self.assertEqual(len(r['events']),1);self.assertEqual(r['events'][0]['duplicate_observations_removed'],1);self.assertEqual((r['events'][0]['start'],r['events'][0]['end']),(1,1.5))
 def test_same_pass_repeated_words_never_merge(self):
  r=run([obs('a',1,1.2),obs('b',1.2,1.4)],['w0','w0']);self.assertEqual(len(r['events']),2)
 def test_separated_okay_remain_separate(self):
  r=run([obs('a',1.78,2.1,'Okay'),obs('b',4.08,4.4,'Okay')],['w0','w3']);self.assertEqual(len(r['events']),2)
 def test_broad_description_cannot_bridge_two_words(self):
  rows=[obs('a',1,1.3),obs('b',1.4,1.7),obs('c',1,1.7,"Says 'yeah'.",word=False)]
  r=run(rows,['w0','w0','w1']);self.assertEqual(len(r['events']),2);self.assertEqual(len(r['evidence_links'][0]['candidate_event_ids']),2)
 def test_complete_link_prevents_transitive_drift(self):
  rows=[obs('a',1,1.5),obs('b',1.15,1.65),obs('c',1.3,1.8)]
  r=run(rows,['w0','w1','w2']);self.assertEqual(len(r['events']),2)
 def test_visual_audio_times_separate_and_unknown_is_not_negative(self):
  r=run([obs('a',1,1.5),obs('b',.8,1.8,'nods',modality='visual',word=False)],['a','v'])
  self.assertEqual(len(r['events']),2);self.assertEqual(len(r['response_groups'][0]['member_event_ids']),2);self.assertIsNone(r['response_groups'][0]['training_mask']);self.assertEqual(r['negative_coverage'],{'visual':[],'audio':[]})
 def test_correction_never_fabricates_bounds_or_drops_evidence(self):
  rows=[obs('a',1,1.5)];correction=dict(replace_observation_ids=['a'],feedback_reference='human',events=[dict(description='Yeah',text='Yeah',event_unit='lexical_response',start=None,end=None,status=dict(content='human_confirmed',count='human_confirmed',boundary='unknown'))])
  r=integrate(dict(sample_id='s',duration_s=10,events=rows),decisions(rows,['w0']),[],correction)
  self.assertIsNone(r['events'][0]['start']);self.assertEqual(r['evidence_links'][0]['observation_id'],'a');self.assertFalse(r['events'][0]['training']['precise_boundary_loss'])
if __name__=='__main__':unittest.main()
