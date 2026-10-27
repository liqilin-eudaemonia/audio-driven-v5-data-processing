"""Regression boundaries: speech without words and vocal acoustics cannot hard-reject."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_audio_quality_v2 import event_decision, retriage, row_decision


class TriageTests(unittest.TestCase):
    def event(self,human=.7,noise=.01,words=False):
        return dict(word_support=words,features=dict(rms_dbfs=-25,peak_dbfs=-10,
            maximum_frame_rise_db=35,crest_db=16,kurtosis=10,top5_bins_fraction=.7,
            flatness=.001,low1000=.95,low150=.1),decision=dict(human_score=human,
            speech_score=human,noise_score=noise,top_events=[]))

    def test_vocal_acoustics_with_missing_words_do_not_reject(self):
        for words in (True,False):
            self.assertEqual(event_decision(self.event(words=words))['status'],'keep')

    def test_conflicting_evidence_requires_review(self):
        self.assertEqual(event_decision(self.event(human=.08,noise=.02))['status'],'review')

    def test_noise_agreement_is_suspicion_not_hard_rejection(self):
        self.assertEqual(event_decision(self.event(human=.03,noise=.8))['status'],'suspect_noise')

    def test_dense_unknown_activity_is_review_not_noise(self):
        self.assertEqual(row_decision({'listener':{'status':'keep'},'speaker':{'status':'keep'}},
            {'deferred_reasons':['listener_activity_too_dense']}),'review')

    def test_retriage_preserves_historical_evidence(self):
        original=dict(events=[self.event()],reasons=['old_reject'])
        result=retriage(original)
        self.assertEqual(result['status'],'keep')
        self.assertEqual(original['reasons'],['old_reject'])
        self.assertEqual(result['legacy_qc1_reasons'],['old_reject'])


if __name__=='__main__':unittest.main()
