import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.chinese_asr_filter import should_remove,filter_labels

def event(text,eid='a',a=.2,b=.4):
    return dict(event_id=eid,modality='audio',text=text,description=text,start=a,end=b,
                description_source={'kind':'ASR_original_text'},intervals=[dict(start=a,end=b)])

def label(events):
    return dict(events=events,excluded_audio_events=[],audio_event_alignment=[dict(event_id=e['event_id'],method='model_intersection',intervals=e['intervals']) for e in events if e['modality']=='audio'])

class ChineseFilterTests(unittest.TestCase):
    def test_single_char_and_english_preserved(self):
        for t in ['嗯','哈','呵','哎','嗯！','yeah','Um','uh-huh','bonjour']:
            self.assertFalse(should_remove(event(t)),t)
        for t in ['南京地铁','我唔记得咗','不能','哈 哈']:
            self.assertTrue(should_remove(event(t)),t)
    def test_only_asr_not_description_language(self):
        e=event('南京地铁');e['description_source']={'source':'omni_audio'};self.assertFalse(should_remove(e))
        e['modality']='visual';self.assertFalse(should_remove(e))
    def test_mute_exact_interval_with_local_fades_and_audit(self):
        src=label([event('不能'),event('嗯','b',.6,.7)]);before=copy.deepcopy(src)
        result,mask,support=filter_labels(src,dict(sample_rate=1000,samples=1000,positive_intervals_samples=[[0,1000]]))
        self.assertTrue(np.all(mask[200:400]==0));self.assertTrue(np.all(mask[:190]==1));self.assertTrue(np.all(mask[410:]==1))
        self.assertEqual([e['text'] for e in result['events']],['嗯']);self.assertEqual(result['excluded_audio_events'][0]['original_event']['text'],'不能');self.assertEqual(src,before)
    def test_overlapping_other_label_clipped_and_visual_exact(self):
        es=[event('不能'),event('yeah','b',.1,.5),dict(event_id='v',modality='visual',description='点头',start=0,end=1)]
        result,mask,support=filter_labels(label(es),dict(sample_rate=1000,samples=1000,positive_intervals_samples=[[0,1000]]))
        self.assertEqual(len(result['events'][0]['intervals']),2);self.assertEqual(result['events'][1],es[2]);self.assertEqual(result['events'][0]['text'],'yeah')

if __name__=='__main__':unittest.main()
