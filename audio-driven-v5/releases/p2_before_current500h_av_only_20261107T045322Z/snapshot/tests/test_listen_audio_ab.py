import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_audio_ab import SR,analyze,reduce


class AudioABTests(unittest.TestCase):
    def test_quiet_noise_reduced_and_untranscribed_burst_protected(self):
        rng=np.random.default_rng(17);x=rng.normal(0,.001,SR*6).astype(np.float32)
        t=np.arange(SR)/SR
        # A non-speech-like modulated sound has NO VAD or transcript annotation.
        x[SR*2:SR*3]+=(.04*np.sin(2*np.pi*340*t)*(1+.5*np.sin(2*np.pi*7*t))).astype(np.float32)
        prepared=analyze(x,[],[])
        for mode in ('mild','strong'):
            y=reduce(prepared,mode)
            self.assertEqual(len(x),len(y));self.assertTrue(np.isfinite(y).all())
            quiet=slice(SR//2,SR)
            self.assertLess(np.linalg.norm(y[quiet])/np.linalg.norm(x[quiet]),.65)
            event=slice(SR*2+SR//5,SR*3-SR//5)
            self.assertGreater(np.linalg.norm(y[event])/np.linalg.norm(x[event]),.80)
            self.assertEqual(int(np.argmax(np.correlate(x[event][::50],y[event][::50],'full'))),
                             len(x[event][::50])-1)

    def test_silence_remains_finite_and_length_exact(self):
        x=np.zeros(SR*2+127,dtype=np.float32);prepared=analyze(x,[],[])
        for mode in ('mild','strong'):
            y=reduce(prepared,mode)
            self.assertEqual(len(y),len(x));self.assertTrue(np.array_equal(x,y))

    def test_unreliable_noise_reference_does_not_invent_spectral_removal(self):
        x=(.05*np.sin(np.arange(SR*3)*2*np.pi*240/SR)).astype(np.float32)
        p=analyze(x,[[0,3]],[[0,3]])
        self.assertFalse(p['noise_profile_available'])
        for mode in ('mild','strong'):
            self.assertLess(np.max(np.abs(reduce(p,mode)-x)),1e-6)


if __name__=='__main__':unittest.main()
