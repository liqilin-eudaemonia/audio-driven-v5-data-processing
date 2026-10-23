import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import listen_v8_audio_first as policy

class AudioFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'data_previews/seamless_listen_v8_recovery/samples.json').read_text())['samples']
        cls.feedback=json.loads((ROOT/'data_reviews/listener_v8_recovery_feedback_20261023.json').read_text())['rows']
    def test_all_reviewed_dispositions(self):
        for n,row in enumerate(self.rows,1):
            expected='audio_reject' if n in (28,33,34) else 'keep_candidate'
            self.assertEqual(policy.route(row,self.feedback)['pool'],expected,n)
    def test_three_bad_audio_examples_detected_without_human_override(self):
        for n in (28,33,34):self.assertEqual(policy.route(self.rows[n-1],[])['pool'],'audio_reject',n)
    def test_identity_content_and_boundary_uncertainty_is_advisory(self):
        for n in (19,20,23,30,31,32,35,36):
            q=policy.route(self.rows[n-1],[])
            self.assertEqual(q['pool'],'keep_candidate',n);self.assertTrue(q['advisory_reasons'])
    def test_approved_weak_noise_example_needs_exact_override(self):
        row=copy.deepcopy(self.rows[28])
        self.assertEqual(policy.route(row,[])['pool'],'audio_reject')
        self.assertEqual(policy.route(row,self.feedback)['pool'],'keep_candidate')
        row['end']+=.1;row['duration_s']+=.1
        q=policy.route(row,self.feedback)
        self.assertEqual(q['pool'],'audio_reject');self.assertFalse(q['human_exact_acceptance'])
    def test_bad_interval_not_escaped_by_renaming(self):
        row=copy.deepcopy(self.rows[27]);row['sample_id']+='renamed';row['start']+=.1
        q=policy.route(row,self.feedback)
        self.assertEqual(q['pool'],'audio_reject');self.assertIn('human_repair_reserve',q['reasons'])
    def test_audio_precedes_task_and_boundary(self):
        row=copy.deepcopy(self.rows[30]);row['qc4']['audio_reasons'].append('strong_reference_match')
        q=policy.route(row,[])
        self.assertEqual(q['pool'],'audio_reject');self.assertIn('strong_reference_match',q['reasons'])
    def test_confirmed_takeover_not_promoted(self):
        row=copy.deepcopy(self.rows[1]);row['listener_compatibility']['substantive']=[{'text':'formal turn'}]
        self.assertEqual(policy.route(row,[])['pool'],'task_hold')
    def test_explicit_old_human_hold_preserved(self):
        row=copy.deepcopy(self.rows[1]);row['release']['reasons']=['human_role_review']
        self.assertEqual(policy.route(row,self.feedback)['pool'],'human_hold')
    def test_truncated_speaker_words_not_implicitly_accepted(self):
        row=copy.deepcopy(self.rows[1]);row['start']=row['speech_start']+.1
        self.assertEqual(policy.route(row,[])['pool'],'task_hold')

if __name__=='__main__':unittest.main()
