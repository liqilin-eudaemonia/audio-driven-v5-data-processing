import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_v8_release as release

ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'data_previews/seamless_listen_v8_qc4/samples.json').read_text())['samples']
        cls.feedback=json.loads((ROOT/'data_reviews/listener_v8_qc4_feedback.json').read_text())['rows']
    def test_reviewed_24_dispositions(self):
        for i,r in enumerate(self.rows,1):
            want='role_review' if i==15 else 'repair_reserve' if i in (18,20,21,22,24) else 'content_review' if i in (19,23) else 'candidate'
            self.assertEqual(release.route(r,self.feedback)['pool'],want,i)
    def test_approval_does_not_promote_expanded_interval(self):
        r=copy.deepcopy(self.rows[12]);r['start']-=.1
        result=release.route(r,self.feedback)
        self.assertFalse(result['human_exact_acceptance']);self.assertNotEqual(result['pool'],'candidate')
    def test_negative_interval_cannot_be_evaded_by_new_id(self):
        r=copy.deepcopy(self.rows[17]);r['sample_id']+='new';r['start']+=.1
        self.assertEqual(release.route(r,self.feedback)['pool'],'repair_reserve')
    def test_nonoverlapping_clip_is_not_quarantined_by_pair(self):
        r=copy.deepcopy(self.rows[17]);r.update(start=10000,end=10005,audio_qc_status='keep',source_manual_rejections=[])
        r['qc4'].update(task_reasons=[],audio_reasons=[],status='candidate')
        self.assertEqual(release.route(r,self.feedback)['pool'],'candidate')
    def test_unconfirmed_noise_is_reserve_not_accepted(self):
        r=copy.deepcopy(self.rows[0]);r['qc4']['audio_reasons']=['listener_speech_identity_unconfirmed'];r['qc4']['status']='review'
        q=release.route(r,[]);self.assertEqual(q['pool'],'repair_reserve');self.assertFalse(q['repair_authorized'])
    def test_content_and_audio_retains_content_block(self):
        r=copy.deepcopy(self.rows[0]);r['qc4'].update(task_reasons=['listener_content_not_confirmed_backchannel'],audio_reasons=['strong_reference_match'],status='review')
        q=release.route(r,[]);self.assertEqual(q['pool'],'content_review');self.assertEqual(q['secondary_audio_concerns'],['strong_reference_match'])
    def test_previous_explicit_rejection_not_promoted(self):
        r=copy.deepcopy(self.rows[0]);r['source_manual_rejections']=[{'decision':'reject'}]
        self.assertEqual(release.route(r,self.feedback)['pool'],'excluded')

if __name__=='__main__':unittest.main()
