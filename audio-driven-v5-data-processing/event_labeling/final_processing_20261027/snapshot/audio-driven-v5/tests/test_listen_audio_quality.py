import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_audio_quality import find_events,features,decision


class QualityTests(unittest.TestCase):
    def normal(self):
        return dict(rms_dbfs=-25.,peak_dbfs=-15.,crest_db=10.,kurtosis=4.,low150=.01,
            low1000=.4,flatness=.05,top5_bins_fraction=.2,maximum_frame_rise_db=5.)

    def test_click_without_vad_is_still_examined(self):
        x=np.zeros(48000,dtype=np.float32);x[24000]=.5
        events=find_events(x,48000,[])
        self.assertTrue(any(a<=.5<=b for a,b in events))
        self.assertEqual(find_events(np.zeros(48000),48000,[]),[])

    def test_confident_laughter_can_pass_without_words(self):
        r=decision(self.normal(),dict(human_score=.8,noise_score=.01,speech_score=.02), 'listener',False)
        self.assertTrue(r['accepted'])

    def test_unknown_audible_event_is_not_assumed_to_be_a_reaction(self):
        r=decision(self.normal(),dict(human_score=.03,noise_score=.05,speech_score=.01), 'listener',False)
        self.assertFalse(r['accepted'])

    def test_transcript_or_model_speech_does_not_override_listener_artifact(self):
        f=dict(self.normal(),maximum_frame_rise_db=35.,crest_db=16.,kurtosis=10.,top5_bins_fraction=.7)
        r=decision(f,dict(human_score=.8,noise_score=.01,speech_score=.8),'listener',True)
        self.assertFalse(r['accepted'])

    def test_constant_signal_metrics_are_finite(self):
        f=features(np.zeros(4800),48000)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))


if __name__=='__main__':unittest.main()
