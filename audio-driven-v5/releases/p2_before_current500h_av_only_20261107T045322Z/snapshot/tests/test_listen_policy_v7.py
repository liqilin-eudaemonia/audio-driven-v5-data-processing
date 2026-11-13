import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_policy_v7 import natural_clips, boundary_activity
from listen_turns_v7 import annotate


def words(duration):
    return [dict(start=i*.5,end=min(duration,i*.5+.4),text='word.' if i%13==0 else 'word')
            for i in range(int(duration*2))]


class NaturalBoundaryTests(unittest.TestCase):
    def assert_valid(self,clips,ws):
        for c in clips:
            self.assertGreaterEqual(c['duration_s'],2-1e-7)
            self.assertLessEqual(c['duration_s'],10+1e-7)
            for edge in (c['start'],c['end']):
                self.assertFalse(any(w['start']+1e-7<edge<w['end']-1e-7 for w in ws))
        for a,b in zip(clips,clips[1:]):self.assertLessEqual(a['end'],b['start']+1e-7)

    def test_short_whole_turn_and_no_frame_gate(self):
        for duration in (2.,2.5,3.,9.99,10.):
            ws=[dict(start=0.,end=duration,text='whole')]
            clips,bad=natural_clips(0,duration,ws,[(0,duration)],[(0,duration)])
            self.assertEqual(len(clips),1);self.assertFalse(bad)
            self.assertAlmostEqual(clips[0]['duration_s'],duration)

    def test_long_turn_preserves_pauses_and_balances_tail(self):
        for duration in (10.5,11.5,19.5,20.5,31.5,61.5):
            ws=words(duration)
            clips,bad=natural_clips(0,duration,ws)
            self.assert_valid(clips,ws);self.assertFalse(bad)
            self.assertAlmostEqual(sum(c['duration_s'] for c in clips),ws[-1]['end'])
            for a,b in zip(clips,clips[1:]):self.assertAlmostEqual(a['end'],b['start'])

    def test_parent_local_words_do_not_require_global_sentence(self):
        ws=words(30)
        clips,bad=natural_clips(8.2,18.2,ws)
        self.assertTrue(clips);self.assert_valid(clips,ws)
        self.assertGreater(sum(c['duration_s'] for c in clips),8)
        self.assertTrue(any(r['rejection']=='speaker_word_crosses_floor_or_exclusion_edge' for r in bad))

    def test_listener_boundary_does_not_discard_clip(self):
        ws=words(8)
        clips,bad=natural_clips(0,8,ws,(),[(-.2,.4),(7.5,8.5)])
        self.assertEqual(len(clips),1);self.assertFalse(bad)
        flags=boundary_activity(clips[0]['start'],clips[0]['end'],dict(vad=[(-.2,.4),(7.5,8.5)],segments=[]))
        self.assertTrue(flags['onset_crossing']);self.assertTrue(flags['end_crossing'])

    def test_seeded_boundary_properties(self):
        rng=random.Random(29)
        for trial in range(100):
            ws=[];t=0.
            for i in range(rng.randrange(6,120)):
                end=t+rng.uniform(.08,.65)
                ws.append(dict(start=t,end=end,text='word.' if i%11==0 else 'word'))
                t=end+rng.uniform(0,.5)
            clips,bad=natural_clips(0,ws[-1]['end'],ws)
            self.assert_valid(clips,ws)
            self.assertAlmostEqual(sum(c['duration_s'] for c in clips)+sum(r['end']-r['start'] for r in bad),ws[-1]['end'],places=6)

    def test_short_floor_and_carry_in_feedback_survive_upstream(self):
        speaker=dict(id='sp',transcript=[dict(start=1.,end=3.5,transcript='hello there',words=[
            dict(start=1.,end=2.,word='hello'),dict(start=2.1,end=3.5,word='there')])],vad=[dict(start=1.,end=3.5)])
        listener=dict(id='li',transcript=[dict(start=.8,end=1.3,transcript='yeah',words=[
            dict(start=.8,end=1.3,word='yeah')])],vad=[dict(start=.8,end=1.3)])
        ann=annotate(dict(pair='p',subset='naturalistic',sources=[dict(duration=4),dict(duration=4)]),
                     dict(records=[speaker,listener]),defer_untranscribed=True)
        self.assertEqual(len(ann['clips']),1)
        self.assertEqual(ann['clips'][0]['listener_events'][0]['label'],'feedback_candidate')
        self.assertAlmostEqual(ann['clips'][0]['start'],1.)


if __name__=='__main__':unittest.main()
