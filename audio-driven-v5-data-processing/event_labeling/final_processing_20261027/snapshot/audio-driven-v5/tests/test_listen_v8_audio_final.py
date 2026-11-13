import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import listen_v8_audio_final as policy

class FinalAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'data_previews/seamless_listen_v8_audio_first/samples.json').read_text())['samples']
        cls.feedback=json.loads((ROOT/'data_reviews/listener_v8_audio_first_feedback_20261023.json').read_text())['rows']
    def test_missed_bleed_detected_without_human_label(self):
        q=policy.route(self.rows[5],[])
        self.assertEqual(q['pool'],'audio_reject');self.assertIn('repeated_audible_delayed_match',q['reasons'])
    def test_repeat_requires_consistent_delay(self):
        r=copy.deepcopy(self.rows[5]);r['qc3']['listener']['events'][1]['paired']['listener_delay_ms']+=50
        self.assertNotIn('repeated_audible_delayed_match',policy.evidence(r)['hard_reasons'])
    def test_duplicate_event_not_independent_repeat(self):
        r=copy.deepcopy(self.rows[5]);w=r['qc3']['listener']['events'][0];r['qc3']['listener']['events']=[w,copy.deepcopy(w)]
        self.assertNotIn('repeated_audible_delayed_match',policy.evidence(r)['hard_reasons'])
    def test_weak_acceptable_examples_auto_kept(self):
        for n in (18,22,23):self.assertEqual(policy.route(self.rows[n-1],[])['pool'],'keep_candidate',n)
    def test_ambiguous_example_not_silently_promoted(self):
        self.assertEqual(policy.route(self.rows[18],[])['pool'],'audio_review')
        self.assertEqual(policy.route(self.rows[18],self.feedback)['pool'],'keep_candidate')
    def test_reviewed_approval_does_not_extend(self):
        r=copy.deepcopy(self.rows[18]);r['end']+=.1;r['duration_s']+=.1
        self.assertEqual(policy.route(r,self.feedback)['pool'],'audio_review')
    def test_other_bad_comparisons_withheld(self):
        for n in (17,20,21,24):self.assertEqual(policy.route(self.rows[n-1],[])['pool'],'audio_reject',n)
    def test_low_floor_does_not_excuse_loud_noise_burst(self):
        r=copy.deepcopy(self.rows[22]);w=r['qc3']['listener']['background'][0]
        w['panns']['noise_score']=.9;w['features']['rms_dbfs']=-15
        self.assertNotEqual(policy.route(r,[])['pool'],'keep_candidate')
    def test_task_and_human_guards_remain(self):
        r=copy.deepcopy(self.rows[21]);r['listener_compatibility']['substantive']=[{'text':'formal response'}]
        self.assertEqual(policy.route(r,[])['pool'],'task_hold')
        r['release']['reasons']=['human_role_review']
        self.assertEqual(policy.route(r,self.feedback)['pool'],'human_hold')
    def test_evidence_does_not_mutate_inputs(self):
        r=copy.deepcopy(self.rows[5]);before=copy.deepcopy(r);policy.route(r,self.feedback)
        self.assertEqual(r,before)

if __name__=='__main__':unittest.main()
