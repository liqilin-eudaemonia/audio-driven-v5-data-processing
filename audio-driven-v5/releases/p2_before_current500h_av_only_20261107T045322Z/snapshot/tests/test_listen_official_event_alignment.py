import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.official_audio_gate import make_gate
from listen_benchmark.official_event_alignment import align_events

def event(i,a,b,text=None,description=None):
    return dict(event_id=i,start=a,end=b,text=text,description=description or text,modality='audio',categories=['listener_vocalization'])

def align(es,words=(),vad=(),sentences=()):
    official={k:[dict(id=f'{k}-{i}',start=a,end=b,text=t,parent='s0') for i,(a,b,t) in enumerate(xs)] for k,xs in [('words',words),('vad',vad),('sentences',sentences)]}
    _,g=make_gate(official,10000,1000)
    return align_events(es,official,g)

class AlignmentTests(unittest.TestCase):
    def test_word_uses_official_end_not_intersection(self):
        e=event('e',2,2.5,'yeah');r=align([e],[(2.2,2.7,'Yeah.')]);f=r['events'][0]
        self.assertEqual((f['start'],f['end']),(2.2,2.7));self.assertEqual(f['description'],'yeah');self.assertEqual(f['source_interval'],dict(start=2,end=2.5))
    def test_conflict_preserves_our_text(self):
        e=event('e',2,2.5,'Um');r=align([e],[(2.2,2.7,'Yeah.')]);f=r['events'][0]
        self.assertEqual(f['text'],'Um');self.assertEqual(f['boundary_method'],'model_intersection');self.assertTrue(f['boundary_alignment']['text_not_literal_match'])
    def test_two_okays_remain_separate(self):
        es=[event('a',1,1.3,'okay'),event('b',1.6,1.9,'okay')]
        r=align(es,[(1.05,1.35,'Okay.'),(1.65,1.95,'Okay.')])
        self.assertEqual([e['event_id'] for e in r['events']],['a','b']);self.assertEqual([e['start'] for e in r['events']],[1.05,1.65])
    def test_two_events_cannot_claim_one_word(self):
        r=align([event('a',1,1.3,'yeah'),event('b',1.1,1.4,'yeah')],[(1.05,1.35,'Yeah')])
        self.assertTrue(all(e['boundary_method']=='model_intersection' for e in r['events']));self.assertEqual(len(r['events']),2)
    def test_ambiguous_repeated_word_does_not_snap(self):
        r=align([event('a',1.1,1.3,'yeah')],[(1,1.2,'Yeah'),(1.2,1.4,'Yeah')])
        self.assertEqual(r['events'][0]['boundary_method'],'model_intersection')
    def test_phrase_and_hyphen_punctuation(self):
        r=align([event('a',1,1.4,'Uh huh')],[(1.03,1.42,'Uh-huh.')]);self.assertEqual(r['events'][0]['boundary_method'],'official_words')
        r=align([event('a',2,2.6,'All right')],[(2.02,2.2,'All'),(2.24,2.64,'right.')]);self.assertEqual(r['events'][0]['end'],2.64)
    def test_isolated_laughter_uses_vad_without_padding(self):
        r=align([event('a',2,2.6,description='Laughter')],vad=[(2.05,2.65,'')]);f=r['events'][0]
        self.assertEqual(f['boundary_method'],'official_vad');self.assertEqual((f['start'],f['end']),(2.05,2.65))
    def test_shared_vad_does_not_expand_each_event(self):
        r=align([event('a',2,2.3,description='Laughter'),event('b',2.4,2.7,'ha')],vad=[(2,2.7,'')])
        self.assertEqual(r['events'][0]['end'],2.3);self.assertTrue(all(e['boundary_method']=='model_intersection' for e in r['events']))
    def test_disconnected_support_not_continuous(self):
        r=align([event('a',1,4,'original phrase')],vad=[(1.2,1.4,''),(3.2,3.4,'')]);f=r['events'][0]
        self.assertEqual(len(f['intervals']),2);self.assertTrue(f['disjoint_support']);self.assertEqual(r['audio_events_retained'],1)
    def test_no_new_events_and_visual_unchanged(self):
        es=[event('a',1,1.3,'yeah'),dict(event_id='v',modality='visual',start=0,end=4,description='Nods.')];before=copy.deepcopy(es)
        r=align(es,[(3,3.3,'Yeah')]);self.assertEqual(r['events'],[es[1]]);self.assertEqual(len(r['excluded_audio_events']),1);self.assertEqual(es,before)
    def test_sentence_is_support_not_word_boundary(self):
        r=align([event('a',2,2.3,'yeah')],sentences=[(1,4,'Yeah, yeah.')]);self.assertEqual((r['events'][0]['start'],r['events'][0]['end']),(2,2.3))
    def test_official_phrase_does_not_fill_muted_gap(self):
        r=align([event('a',2,2.7,'all right')],[(2,2.2,'All'),(2.48,2.7,'right')]);f=r['events'][0]
        self.assertEqual(f['boundary_method'],'official_words');self.assertEqual(len(f['intervals']),2)
        self.assertLess(f['intervals'][0]['end'],f['intervals'][1]['start'])

if __name__=='__main__':unittest.main()
