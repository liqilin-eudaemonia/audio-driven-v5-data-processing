import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import signal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_audio_quality_v4 as qc


class QualityV4Tests(unittest.TestCase):
    def test_delayed_filtered_reference_outside_old_150ms_window(self):
        sr=8000;rng=np.random.default_rng(8)
        x=signal.sosfilt(signal.butter(3,[200,3000],fs=sr,btype='bandpass',output='sos'),rng.normal(0,.1,sr*5))
        delay=int(.42*sr);y=np.r_[np.zeros(delay),x[:-delay]]*.3+rng.normal(0,.0001,len(x))
        result=qc.reference_scan(x,y,sr)
        self.assertIn('consistent_reference_leakage',result['reasons'])
        matched=[w for w in result['windows'] if w['correlation']>.9]
        self.assertTrue(matched);self.assertAlmostEqual(matched[0]['delay_ms'],420,delta=1)

    def test_independent_activity_does_not_match_reference(self):
        rng=np.random.default_rng(4)
        result=qc.reference_scan(rng.normal(0,.1,32000),rng.normal(0,.02,32000),8000)
        self.assertEqual(result['status'],'keep')

    def test_silent_listener_is_not_echo(self):
        rng=np.random.default_rng(2)
        self.assertEqual(qc.reference_scan(rng.normal(0,.1,16000),np.zeros(16000),8000)['status'],'keep')

    def event(self,laughter=False):
        labels=[dict(label='Speech',score=.8)]+([dict(label='Laughter',score=.7)] if laughter else [])
        return dict(start=.5,end=1.,word_support=False,features=dict(rms_dbfs=-25,peak_dbfs=-10),
            ast=dict(speech_score=.8,top_events=labels),panns=dict(top_events=labels))

    def test_untranscribed_speech_is_review_evidence_not_noise(self):
        row=dict(qc3=dict(listener=dict(events=[self.event()])))
        self.assertEqual(qc.speech_identity(row)[0]['reason'],'listener_speech_identity_unconfirmed')

    def test_untranscribed_laughter_not_rejected_for_missing_text(self):
        self.assertEqual(qc.speech_identity(dict(qc3=dict(listener=dict(events=[self.event(True)])))),[])

    def test_manual_holds_are_local_and_separate_from_automatic_detection(self):
        row=dict(listener_id='listener',start=2.,end=4.,qc3=dict(listener=dict(events=[])),
            listener_compatibility={},v8_combined_status='candidate',audio_qc_status='keep')
        feedback=[dict(listener_id='listener',start=3.,end=3.5,sample_id='manual')]
        with patch.object(qc,'reference_scan',return_value=dict(reasons=[])):
            result=qc.assess(row,[],[],feedback)
            self.assertEqual(result['automatic_status'],'candidate');self.assertEqual(result['status'],'held_by_human')
            row.update(start=5.,end=6.)
            self.assertEqual(qc.assess(row,[],[],feedback)['status'],'candidate')

    def test_short_content_is_not_automatically_a_backchannel(self):
        row=dict(listener_id='l',start=0.,end=4.,qc3=dict(listener=dict(events=[])),
            listener_compatibility=dict(uncertain_short_speech=[dict(text='Everybody be hungry.')]),
            v8_combined_status='candidate',audio_qc_status='keep')
        with patch.object(qc,'reference_scan',return_value=dict(reasons=[])):
            self.assertEqual(qc.assess(row,[],[])['automatic_status'],'review')


if __name__=='__main__':unittest.main()
