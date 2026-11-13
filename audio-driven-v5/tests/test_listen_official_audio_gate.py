import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.official_audio_gate import make_gate,gate_audio,adapt_events

def official(words=(),sentences=(),vad=()):
 return {k:[dict(start=a,end=b) for a,b in v] for k,v in [('words',words),('sentences',sentences),('vad',vad)]}
class GateTests(unittest.TestCase):
 def test_three_sources_union_padding_merge(self):
  m,i=make_gate(official(words=[(.2,.3)],sentences=[(.5,.6)],vad=[(.8,.9)]),1000,1000)
  self.assertEqual(i['keep_intervals_samples'],[[100,1000]]);self.assertTrue(np.all(m[:100]==0));self.assertTrue(np.all(m[200:900]==1))
 def test_empty_and_outside_neighbor_are_silent(self):
  m,i=make_gate(official(words=[(1.01,1.1)]),1000,1000)
  self.assertFalse(m.any());self.assertTrue(i['full_silence']);self.assertEqual(i['skipped_official_intervals'][0]['reason'],'outside_clip')
 def test_clip_before_padding_and_no_clip_edge_fade(self):
  m,i=make_gate(official(vad=[(-1,.1),(.9,2)]),1000,1000)
  self.assertEqual(m[0],1);self.assertEqual(m[-1],1);self.assertTrue(np.all(m[200:800]==0))
 def test_fade_is_smooth_and_core_unchanged(self):
  m,i=make_gate(official(vad=[(.4,.6)]),1000,1000)
  self.assertEqual(m[300],0);self.assertEqual(m[309],1);self.assertTrue(np.all(np.diff(m[300:310])>=0));self.assertTrue(np.all(m[400:600]==1));self.assertEqual(m[699],0)
 def test_overlap_not_double_attenuated(self):
  m,i=make_gate(official(words=[(.3,.5),(.4,.6)]),1000,1000)
  self.assertEqual(i['keep_intervals_samples'],[[200,700]]);self.assertTrue(np.all(m[300:600]==1))
 def test_waveform_stereo_and_immutability(self):
  x=np.ones((1000,2),dtype=np.float32)*.5;before=x.copy();z,m,i=gate_audio(x,official(vad=[(.4,.6)]),1000)
  self.assertEqual(z.shape,x.shape);np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(z[m==1],x[m==1]);self.assertTrue(np.all(z[m==0]==0))
 def test_labels_partial_excluded_visual(self):
  mask,_=make_gate(official(vad=[(.2,.3)]),1000,1000,padding_seconds=0,fade_seconds=0)
  es=[dict(event_id='partial',modality='audio',start=.1,end=.5,description='original Yeah'),dict(event_id='muted',modality='audio',start=.6,end=.7,description='x'),dict(event_id='negative',modality='audio',start=-.3,end=-.2,description='x'),dict(event_id='visual',modality='visual',start=0,end=1,description='nod')]
  r=adapt_events(es,mask,1000);self.assertEqual(len(r['excluded_audio_events']),2);self.assertEqual(r['events'][0]['description'],'original Yeah');self.assertEqual(r['events'][0]['start'],.1);self.assertEqual(r['events'][0]['audio_gate']['retained_intervals'],[dict(start=.2,end=.3)]);self.assertEqual(r['events'][1],es[-1])
 def test_invalid_input_does_not_silently_become_empty(self):
  for data in [{},official(vad=[(.8,.2)]),official(vad=[(float('nan'),.2)])]:
   with self.assertRaises(ValueError):make_gate(data,1000,1000)
 def test_zero_length_recorded(self):
  m,i=make_gate(official(words=[(.2,.2)]),1000,1000);self.assertFalse(m.any());self.assertEqual(i['skipped_official_intervals'][0]['reason'],'zero_length')
if __name__=='__main__':unittest.main()
