import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_policy_v8 import sentence_units,group_units,acoustic_cut,quiet_point
from listen_turns_v8 import feedback,prepare


def words(text,start=1.,step=.4):
    return [dict(start=start+i*step,end=start+i*step+.2,text=t) for i,t in enumerate(text.split())]


def person(ws,segments=()):
    return dict(words=ws,segments=list(segments),vad=[(w['start'],w['end']) for w in ws])


class CompleteSentenceTests(unittest.TestCase):
    def test_composed_feedback_does_not_strip_content(self):
        self.assertTrue(feedback('Yeah, makes sense.',1.2))
        self.assertTrue(feedback('Oh, I see.',1.2))
        self.assertFalse(feedback('Yeah, I want to change jobs.',2.))

    def test_coarse_segment_is_not_actual_word_duration(self):
        r=dict(transcript=[dict(start=1,end=5,transcript='Yeah.',words=[dict(start=2,end=2.3,word='Yeah.')])],vad=[])
        s=prepare(r,0)['segments'][0]
        self.assertTrue(s['feedback']);self.assertEqual((s['start'],s['end']),(2,2.3))

    def test_global_prefix_retained(self):
        ws=words("So it's because my body's not used to that.")
        u=sentence_units(ws)
        self.assertEqual(len(u),1);self.assertTrue(u[0]['text'].startswith("So it's"))

    def test_terminal_demonstrative_is_a_valid_sentence_end(self):
        units=sentence_units(words('I like that. Yeah.'))
        self.assertEqual(len(units),2)

    def test_elongated_feedback_is_candidate_with_timing_flag(self):
        r=dict(transcript=[dict(start=1,end=4.6,transcript='Yeah.',words=[dict(start=1,end=4.6,word='Yeah.')])],vad=[])
        s=prepare(r,0)['segments'][0]
        self.assertTrue(s['feedback']);self.assertTrue(s['long_feedback_timing_uncertain'])

    def test_tight_next_voice_margin_does_not_delete_final_word(self):
        ws=words('This is a whole sentence.');u=sentence_units(ws)[0]
        extra=dict(start=u['end']+.1,end=u['end']+.3,text='Next.')
        pp=[person(ws+[extra]),person([])];rms=[np.full(1000,.1),np.full(1000,1e-5)]
        cut,error=acoustic_cut(u,pp,rms,0,10)
        self.assertIsNone(error);self.assertGreaterEqual(cut['end']+1e-6,u['end'])
        self.assertLessEqual(cut['end'],extra['start'])
        self.assertTrue(cut['boundary_review_flags'])

    def test_long_sentence_allowed_without_wordgap_split(self):
        ws=words('We can keep this entire sentence even if it takes longer.',step=1.)
        units=sentence_units(ws);clips,bad=group_units(units,person([]))
        self.assertEqual(len(clips),1);self.assertFalse(bad)
        self.assertGreater(clips[0]['end']-clips[0]['start'],10)

    def test_overlong_sentence_is_explicit_unresolved_not_half_sentence(self):
        ws=words('We can keep this entire sentence even if it takes longer.',step=1.2)
        clips,bad=group_units(sentence_units(ws),person([]))
        self.assertFalse(clips);self.assertEqual(bad[0]['reason'],'complete_unit_exceeds_11_95s')

    def test_inferred_clause_does_not_claim_confirmation(self):
        ws=words('Well, The concept makes sense.',step=.8)
        u=sentence_units(ws)
        self.assertEqual(u[1]['start_evidence'],'inferred_clause_start')
        self.assertIn('inferred_clause_start',u[1]['semantic_flags'])

    def test_groups_never_overlap_or_drop_partial_words(self):
        ws=words('This is sentence one. This is sentence two. This is sentence three.',step=.8)
        groups,bad=group_units(sentence_units(ws),person([]))
        self.assertFalse(bad)
        self.assertEqual([w for g in groups for w in g['words']],ws)
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(groups,groups[1:])))

    def test_formal_listener_speech_blocks_unit(self):
        ws=words('This is a whole sentence.');s=dict(start=2,end=5,text='A substantive answer with many content words here',id='l',feedback=False,word_count=9)
        groups,bad=group_units(sentence_units(ws),person([], [s]))
        self.assertFalse(groups);self.assertTrue(bad)

    def test_short_formal_takeover_is_not_included_as_backchannel(self):
        ws=words('This is a whole sentence.');s=dict(start=2,end=2.5,text='My answer.',id='l',feedback=False,word_count=2,formal_takeover=True)
        groups,bad=group_units(sentence_units(ws),person([], [s]))
        self.assertFalse(groups);self.assertTrue(bad)

    def test_acoustic_padding_and_feedback_tail(self):
        ws=words('This is a whole sentence.');u=sentence_units(ws)[0];b=u['end']
        lw=[dict(start=b-.1,end=b+.35,text='Yeah.')]
        lp=person(lw,[dict(start=b-.1,end=b+.35,text='Yeah.',id='l',feedback=True,word_count=1)])
        pp=[person(ws),lp];rms=[np.full(1000,1e-5),np.full(1000,1e-5)]
        for ch,p in enumerate(pp):
            for w in p['words']:rms[ch][int(w['start']*100):int(w['end']*100)]=.1
        cut,error=acoustic_cut(u,pp,rms,0,10)
        self.assertIsNone(error);self.assertLess(cut['start'],ws[0]['start'])
        self.assertGreater(cut['end'],lw[0]['end']);self.assertLess(cut['end'],b+1.5)

    def test_no_quiet_is_review_not_false_success(self):
        ws=words('This is a whole sentence.');pp=[person(ws),person([])]
        cut,error=acoustic_cut(sentence_units(ws)[0],pp,[np.full(1000,.1)]*2,0,10)
        self.assertIsNone(error);self.assertIn('start_no_quiet_cut',cut['boundary_review_flags'])

    def test_feedback_before_previous_clip_end_is_unresolved(self):
        ws=words('This is a whole sentence.',start=3.);u=sentence_units(ws)[0]
        lw=[dict(start=2.8,end=3.4,text='Yeah.')]
        lp=person(lw,[dict(start=2.8,end=3.4,text='Yeah.',id='l',feedback=True,word_count=1)])
        for energy in (1e-5,.1):
            with self.subTest(energy=energy):
                cut,error=acoustic_cut(u,[person(ws),lp],[np.full(1000,energy)]*2,
                    0,10,left_bound=2.9)
                self.assertIsNone(cut)
                self.assertEqual(error,'complete_content_overlaps_previous_clip')

    def test_feedback_at_previous_clip_end_remains_complete(self):
        ws=words('This is a whole sentence.',start=3.);u=sentence_units(ws)[0]
        lw=[dict(start=2.8,end=3.4,text='Yeah.')]
        lp=person(lw,[dict(start=2.8,end=3.4,text='Yeah.',id='l',feedback=True,word_count=1)])
        cut,error=acoustic_cut(u,[person(ws),lp],[np.full(1000,.1)]*2,0,10,left_bound=2.8)
        self.assertIsNone(error);self.assertEqual(cut['start'],2.8)
        self.assertGreaterEqual(cut['end'],u['end'])

    def test_quiet_requires_sustained_interval(self):
        times=(np.arange(100)+.5)*.01;levels=np.zeros((100,2));levels[40:44]=-80
        self.assertIsNone(quiet_point(times,levels,np.array([-50,-50]),.3,.6,.42))

    def test_listener_background_is_not_used_as_speech_reference(self):
        ws=words('This is a whole sentence.');pp=[person(ws),person([])]
        rms=[np.full(1000,1e-5),np.full(1000,1e-4)]
        for w in ws:rms[0][int(w['start']*100):int(w['end']*100)]=.1
        cut,error=acoustic_cut(sentence_units(ws)[0],pp,rms,0,10)
        self.assertIsNone(error)
        self.assertNotIn('start_no_quiet_cut',cut['boundary_review_flags'])
        self.assertNotIn('end_no_quiet_cut',cut['boundary_review_flags'])


if __name__=='__main__':unittest.main()
