"""Behavioral checks for the user's pause/feedback/interruption requirements."""
import unittest
from build_seamless_rule_turns import annotate, feedback, prepare, track_floors


def segment(start,text,duration=3):
    ts=text.split();step=duration/len(ts)
    return dict(start=start,end=start+duration,transcript=text,
                words=[dict(word=t,start=start+i*step,end=start+i*step+min(.25,step*.9)) for i,t in enumerate(ts)])


def record(person,segments):
    return dict(id=f'test_P{person}',transcript=segments,vad=[dict(start=s['start'],end=s['end']) for s in segments])


class FloorTests(unittest.TestCase):
    def test_pause_and_feedback_do_not_end_floor(self):
        a=record(0,[segment(0,'I remember a long story about our school',4),segment(20,'We kept practicing until the show was finally ready',4)])
        b=record(1,[segment(12,'Yeah.',.3),segment(30,'I remember doing something very similar when I was young',5)])
        turns,_=track_floors([prepare(a,0),prepare(b,1)],40)
        self.assertEqual([t['start'] for t in turns],[0,30])

    def test_yielded_takeover_can_contain_its_own_pause(self):
        a=record(0,[segment(0,'Tell me about your greatest dream',3)])
        b=record(1,[segment(8,'My most profound dream',1.6),segment(20,'is to tell stories that really connect with other people',4)])
        turns,_=track_floors([prepare(a,0),prepare(b,1)],30)
        self.assertEqual([t['start'] for t in turns],[0,8])

    def test_short_interruption_is_excluded_without_concatenation(self):
        a=record(0,[segment(0,'I was walking through the old school with my friends',4),segment(14,'And then we saw something very interesting in the classroom',4)])
        b=record(1,[segment(10,'Wait a minute.',.8)])
        ann=annotate(dict(pair='test',subset='test',sources=[dict(duration=20)]),dict(records=[a,b]))
        self.assertEqual(len(ann['turns']),1)
        events=ann['turns'][0]['events'];self.assertTrue(events[0]['exclude'])
        self.assertEqual(len(ann['clips']),2)
        self.assertLess(ann['clips'][0]['end'],10)
        self.assertGreater(ann['clips'][1]['start'],10.8)

    def test_short_question_is_separate_event(self):
        a=record(0,[segment(0,'Let me explain what happened to us yesterday',4),segment(12,'I mean the other room next to the main entrance',4)])
        b=record(1,[segment(10,'What do you mean?',1)])
        ann=annotate(dict(pair='test',subset='test',sources=[dict(duration=20)]),dict(records=[a,b]))
        self.assertEqual(len(ann['turns']),1)
        self.assertEqual(ann['turns'][0]['events'][0]['label'],'clarification_candidate')

    def test_feedback_prefix_does_not_mask_content(self):
        self.assertTrue(feedback('Oh, yeah.',.6))
        self.assertFalse(feedback('Yeah, but I have another idea.',1.5))


if __name__=='__main__':unittest.main()
