import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_policy_v6 import natural_clips, listener_screen, orientation, require_upright,reviewed_noise_hits


class PolicyTests(unittest.TestCase):
    def test_human_noise_rejection_follows_source_time_not_window_id(self):
        reviews=[dict(listener_id='person_a',start=5.,end=8.,decision='reject',sample_id='old_window')]
        self.assertTrue(reviewed_noise_hits('person_a',7.,9.,reviews))
        self.assertTrue(reviewed_noise_hits('person_a',6.,7.,reviews))
        self.assertFalse(reviewed_noise_hits('person_b',6.,7.,reviews))
        self.assertFalse(reviewed_noise_hits('person_a',8.,10.,reviews))

    def words(self, n=20, punctuation=(9,19)):
        return [dict(start=i*.5+.1,end=i*.5+.4,text='word.' if i in punctuation else 'word') for i in range(n)]

    def test_complete_sentence_and_nonoverlapping_alignment(self):
        ws=self.words(30,(9,19,29))
        clips,_=natural_clips(0,15,ws)
        self.assertTrue(clips)
        for c in clips:
            self.assertLessEqual(c['duration_s'],9.96)
            self.assertEqual((c['frames']-1)%8,0)
            self.assertTrue(c['text'].endswith('.'))
            self.assertGreaterEqual(c['end'],c['speech_end'])
            self.assertLessEqual(c['start'],c['speech_start'])
            self.assertLess(c['alignment_silence_seconds'],.32+1e-7)
            for w in ws:
                self.assertFalse(w['start'] < c['start'] < w['end'])
                self.assertFalse(w['start'] < c['end'] < w['end'])
        for a,b in zip(clips,clips[1:]): self.assertLessEqual(a['end'],b['start'])

    def test_no_hard_cut_long_sentence(self):
        clips,rejected=natural_clips(0,20,self.words(40,(39,)))
        self.assertEqual(clips,[])
        self.assertTrue(rejected)

    def test_sentence_end_beats_later_mid_sentence_pause(self):
        ws=self.words(20,(9,19))
        # A later .7s pause does not take priority over the earlier sentence end.
        for w in ws[16:]:w['start']+=.4;w['end']+=.4
        clips,_=natural_clips(0,11,ws)
        self.assertTrue(clips)
        self.assertEqual(clips[0]['speech_end'],ws[9]['end'])
        self.assertEqual(clips[0]['end_reason'],'sentence_end')

    def test_pause_after_and_is_not_a_clause_end(self):
        ws=self.words(40,())
        ws[14]['text']='and'
        for w in ws[15:]:w['start']+=.6;w['end']+=.6
        clips,_=natural_clips(0,21,ws)
        self.assertEqual(clips,[])

    def test_no_partial_floor_sentence(self):
        clips,_=natural_clips(2,7,self.words())
        self.assertEqual(clips,[])

    def test_no_frame_alignment_inside_vad(self):
        clips,_=natural_clips(0,10,self.words(),speaker_vad=[[0,10]])
        self.assertEqual(clips,[])

    def test_reported_repeated_vad_only_rejected(self):
        p=dict(segments=[],vad=[[16.002,16.478],[17.922,18.302],[18.818,19.710],[20.546,20.590]])
        self.assertFalse(listener_screen(p,14.9695,20.590)['accepted'])

    def test_short_feedback_retained(self):
        p=dict(vad=[[2,2.4]],segments=[dict(start=2,end=2.4,feedback=True,
            words=[dict(start=2,end=2.4)])])
        self.assertTrue(listener_screen(p,0,6)['accepted'])
        p['segments'][0]['feedback']=False
        self.assertFalse(listener_screen(p,0,6)['accepted'])

    def test_raw_orientation_must_be_upright(self):
        k=np.zeros((100,133,3));k[:,:,2]=1
        k[:,5,:2]=[400,600];k[:,6,:2]=[600,600];k[:,0,:2]=[500,400]
        self.assertEqual(require_upright(k,1000,1000)['clockwise_degrees'],0)
        for degrees in (90,180,270):
            t=np.deg2rad(degrees);r=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
            rotated=k.copy();rotated[:,:,:2]=(k[:,:,:2]-500)@r.T+500
            self.assertNotEqual(orientation(rotated,1000,1000)['clockwise_degrees'],0)
            with self.assertRaisesRegex(ValueError,'not_upright'):require_upright(rotated,1000,1000)
        with self.assertRaisesRegex(ValueError,'not_upright'):require_upright(k,1000,1000,90)


if __name__=='__main__':unittest.main()
