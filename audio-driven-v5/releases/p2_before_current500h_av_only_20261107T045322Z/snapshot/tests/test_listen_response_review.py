import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.response_review import alignment_track,deduplicate_tracks

def track(pid,words,full=True):
    return alignment_track(dict(sample_id=pid,words=[dict(text=t,start_time=a,end_time=b) for t,a,b in words],input_evidence={} if full else {'window':{'start_s':0,'end_s':3}}),3)

class ReviewTests(unittest.TestCase):
    def test_touching_same_pass_repetitions_remain_distinct(self):
        a=track('full',[('yeah',.4,.6),('yeah',.6,.8)])
        b=track('window',[('yeah',.4,.6),('yeah',.6,.8)],False)
        out=deduplicate_tracks([a,b]);self.assertEqual(len(out),2);self.assertEqual([len(c['members']) for c in out],[2,2])
    def test_zero_length_token_and_order_survive(self):
        t=track('full',[('yeah',.4,.6),('yeah',.6,.6),('yeah',.6,.6)])
        out=deduplicate_tracks([t]);self.assertEqual(len(out),3);self.assertEqual([c['representative']['token_index'] for c in out],[0,1,2]);self.assertFalse(t['tokens'][1]['time_valid'])
    def test_broad_record_does_not_bridge_two_words(self):
        a=track('full',[('yeah',.4,.7),('yeah',.7,1.)]);b=track('window',[('yeah',.4,1.)],False)
        self.assertEqual(len(deduplicate_tracks([a,b])),3)
    def test_nontransitive_matching_prevents_drift(self):
        ts=[track('full',[('yeah',0.,.4)]),track('w1',[('yeah',.15,.55)],False),track('w2',[('yeah',.3,.7)],False)]
        self.assertEqual([len(c['members']) for c in deduplicate_tracks(ts)],[2,1])
    def test_window_offset_and_invalid_bounds_preserved(self):
        t=alignment_track(dict(sample_id='w',input_evidence={'window':{'start_s':2.,'end_s':3.}},words=[dict(text='yeah',start_time=.5,end_time=1.2)]),4)
        self.assertEqual(t['tokens'][0]['start'],2.5);self.assertEqual(t['tokens'][0]['end'],3.2);self.assertFalse(t['tokens'][0]['time_valid'])
    def test_same_word_long_gap_stays_separate(self):
        a=track('full',[('yeah',0.,.4)]);b=track('w',[('yeah',2.,2.4)],False)
        self.assertEqual(len(deduplicate_tracks([a,b])),2)

if __name__=='__main__':unittest.main()
