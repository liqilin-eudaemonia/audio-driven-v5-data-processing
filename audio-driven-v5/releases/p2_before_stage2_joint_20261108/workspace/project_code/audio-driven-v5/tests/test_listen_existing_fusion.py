import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.existing_result_fusion import select_path,transcript_units,fuse,group_candidates
from listen_benchmark.response_review import alignment_track

def track(pid,text,words,window=None):
 return alignment_track(dict(sample_id=pid,text=text,words=[dict(text=t,start_time=a,end_time=b) for t,a,b in words],input_evidence={} if window is None else {'window':window}),10)
def c(text,start,end,score=2):return dict(text=text,start=start,end=end,score=score)
class ExistingTests(unittest.TestCase):
 def test_endpoint_roundoff_keeps_word_and_original_evidence(self):
  duration=5.72
  t=alignment_track(dict(sample_id='s__w',text='sim',words=[dict(text='sim',start_time=.04,end_time=.12)],input_evidence={'window':dict(start_s=5.6000000000000005,end_s=duration)}),duration)
  reported=t['tokens'][0]['end'];self.assertGreater(reported,duration)
  r=fuse(dict(sample_id='s',duration_s=duration,events=[]),[],[t]);e=r['events'][0]
  self.assertEqual(e['end'],duration);self.assertEqual(e['description'],'sim')
  self.assertEqual(e['timing_evidence']['reported_interval'][1],reported)
  self.assertEqual(t['tokens'][0]['end'],reported)
 def test_real_outside_boundary_is_not_clamped(self):
  t=alignment_track(dict(sample_id='s',text='yeah',words=[dict(text='yeah',start_time=1,end_time=2.01)]),2)
  r=fuse(dict(sample_id='s',duration_s=2,events=[]),[],[t]);self.assertEqual(r['events'],[])
 def test_repetitions_keep_order_and_use_only_existing_intervals(self):
  units=[dict(text='yeah',tokens=[]) for _ in range(3)];options=[c('yeah',1,1.2),c('yeah',1.2,1.4),c('yeah',1.4,1.6),c('yeah',1,1.6,10)]
  path=select_path(units,options);self.assertEqual([(x['start'],x['end']) for x in path],[(1,1.2),(1.2,1.4),(1.4,1.6)])
 def test_missing_boundaries_not_evenly_split(self):
  path=select_path([dict(text='yeah',tokens=[]) for _ in range(3)],[c('yeah',1,2)])
  self.assertEqual(sum(x is not None for x in path),1)
 def test_multiword_response_unit_is_not_two_responses(self):
  t=track('sample','Uh huh. Yeah.',[('Uh',3.5,3.8),('huh',3.8,3.8),('Yeah',8.2,8.4)])
  self.assertEqual([u['text'] for u in transcript_units(t)],['Uh huh','Yeah'])
 def test_chinese_repeated_tokens_not_collapsed(self):
  t=track('sample','嗯嗯。',[('嗯',.3,1.4),('嗯',1.4,1.5)])
  self.assertEqual(len(transcript_units(t)),2)
 def test_zero_only_alignment_becomes_explicit_failure(self):
  t=track('sample','嗯。',[('嗯',.3,.3)])
  result=fuse(dict(sample_id='sample',duration_s=10,events=[]),[],[t])
  self.assertEqual(result['events'],[]);self.assertEqual(len(result['automatic_fusion']['alignment_failures']),1)
 def test_sample_name_does_not_change_generation(self):
  t=track('pass','Okay. Okay.',[('Okay',1,1.4),('Okay',4,4.4)])
  a=fuse(dict(sample_id='first',duration_s=10,events=[]),[],[t]);b=fuse(dict(sample_id='renamed',duration_s=10,events=[]),[],[t]);self.assertEqual(a['events'],b['events'])
 def test_conflicting_local_word_does_not_replace_reference_text(self):
  full=track('full','Foo.',[('Foo',0,5)])
  local=track('window','Bar.',[('Bar',1,1.2)],{'start_s':1,'end_s':4})
  r=fuse(dict(sample_id='sample',duration_s=10,events=[]),[],[full,local])
  self.assertEqual(r['events'][0]['text'],'Foo')
  self.assertEqual(r['events'][0]['description'],'Foo')
  self.assertFalse(r['automatic_fusion']['content_substitutions'])
  self.assertIn('Bar',[c['text'] for c in r['automatic_fusion']['candidate_intervals']])
 def test_laughter_evidence_does_not_reinterpret_asr_words(self):
  obs=[dict(event_id='v',modality='visual',behavior='Laughs.',start=.1,end=2,categories=['laugh'],mapping_status='mapped_candidate'),
       dict(event_id='a',modality='audio',behavior='Laughs.',start=.1,end=2,categories=['listener_vocalization'],mapping_status='mapped_candidate')]
  decisions=[dict(open_event_id='a',origin=dict(signature='laugh'))]
  t=track('full','嘿嘿。',[('嘿',.3,.5),('嘿',.5,.8)])
  r=fuse(dict(sample_id='sample',duration_s=10,events=obs),decisions,[t])
  self.assertEqual([e['description'] for e in r['events'] if e.get('text')],['嘿','嘿'])
  self.assertFalse(r['automatic_fusion']['phonetic_laughter_links'])
  self.assertTrue(any(e['description']=='Laughs.' for e in r['events']))
 def test_model_description_is_restored_verbatim(self):
  raw='The listener smiles briefly, then looks down.'
  obs=[dict(event_id='v',modality='visual',behavior='Smiles.',start=1,end=2,categories=['smile'],mapping_status='mapped_candidate',source_descriptions=[dict(description=raw,source='visual_model')])]
  r=fuse(dict(sample_id='sample',duration_s=10,events=obs),[],[])
  self.assertEqual(r['events'][0]['description'],raw)
  self.assertEqual(r['events'][0]['original_descriptions'],[raw])
  self.assertEqual(obs[0]['behavior'],'Smiles.')
 def test_builder_and_algorithm_do_not_contain_regression_sample_ids(self):
  root=Path(__file__).resolve().parents[1]
  for p in [root/'src/listen_benchmark/existing_result_fusion.py',root/'scripts/build_listen_existing_fusion.py']:
   s=p.read_text()
   for forbidden in ['S0805','S0835','S2043','S3358','S5193','torch','qwen_asr','soundfile']:self.assertNotIn(forbidden,s)
if __name__=='__main__':unittest.main()
