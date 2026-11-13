import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_audio_quality_v3 import paired_similarity,decide_event,background_decision,full_coverage_windows


class PairedQualityTests(unittest.TestCase):
    def f(self):return dict(rms_dbfs=-25,peak_dbfs=-12,flatness=.05)
    def test_lagged_copy_detected_without_amplitude_dependence(self):
        rng=np.random.default_rng(123);x=rng.normal(size=16000)*.1
        y=np.r_[np.zeros(80),x[:-80]]*.15
        p=paired_similarity(x,y,.2,.8)
        self.assertGreater(p['abs_wave_correlation'],.999)
        self.assertAlmostEqual(p['listener_delay_ms'],5.,places=3)
    def test_independent_simultaneous_sounds_are_not_copies(self):
        rng=np.random.default_rng(0);x=rng.normal(size=16000)*.1;y=rng.normal(size=16000)*.1
        p=paired_similarity(x,y,.2,.8)
        self.assertLess(p['abs_wave_correlation'],.1)
    def test_human_evidence_does_not_cancel_environmental_noise(self):
        ast=dict(human_score=.9,top_events=[dict(label='Static',score=.8)])
        result=decide_event(self.f(),ast,dict(human_score=.9,noise_score=.7),dict(available=False))
        self.assertEqual(result['status'],'exclude')
    def test_untranscribed_reaction_can_pass(self):
        result=decide_event(self.f(),dict(human_score=.7,top_events=[]),dict(human_score=.8,noise_score=.01),dict(available=False))
        self.assertEqual(result['status'],'keep')
    def test_background_scan_covers_noninteger_tail(self):
        windows=full_coverage_windows(9.64);self.assertEqual(windows[0][0],0)
        self.assertEqual(windows[-1][1],9.64)
        self.assertTrue(all(windows[i+1][0]<=windows[i][1] for i in range(len(windows)-1)))
    def test_background_noise_stays_flagged_during_human_speech(self):
        result=background_decision(self.f(),dict(human_score=.95,noise_score=.5))
        self.assertEqual(result['status'],'review')

if __name__=='__main__':unittest.main()
