import copy
import unittest
from unittest.mock import patch
import promotion_policy as policy
from p2_event_presence.contracts import SCHEMA, labels


def fixtures(good=True):
    membership=[]; events=[]; video=[]; audio=[]
    for i in range(24):
        sid=str(i); active=i%2==0
        membership.append(dict(id=sid,group_id=str(i//2)))
        gt=labels('present' if active else 'absent')
        pred=copy.deepcopy(gt) if good else labels('absent')
        events.append(dict(schema=SCHEMA,dataset='seamless',id=sid,method='model',draw_id='0',
            reference_identity='media'+sid,protocol_identity='detector',capabilities=dict(visual=True,audio=True),
            reference=gt,generated=pred))
        video.append(dict(id=sid,metrics=dict(lpips=.2,psnr=20.,ssim=.7)))
        audio.append(dict(id=sid,errors={},audio=dict(comparison=dict(acoustic=dict(
            any_reference_activity=active,false_activity_on_silent_clip=False,
            missed_entire_response=active and not good)))))
    return membership,dict(events=events,video=video,audio=audio)


class PromotionChecks(unittest.TestCase):
    def decision(self,data,step=500,previous=None):
        membership,_=fixtures()
        with patch.dict(policy.DEFAULT,bootstrap_draws=1000):
            return policy.decide(data,membership,copy.deepcopy(policy.DEFAULT),step,previous)

    def test_clearly_dominated_candidate_can_be_eliminated(self):
        _,good=fixtures();_,bad=fixtures(False)
        d=self.decision(dict(baseline=good,event100=bad))
        self.assertEqual(d['candidates'],['baseline']);self.assertEqual(d['eliminated'],['event100'])
        self.assertEqual(d['point_metrics']['baseline']['reaction'],1.)
        self.assertEqual(d['point_metrics']['event100']['reaction'],0.)

    def test_equal_results_retain_all_no_top_k(self):
        _,good=fixtures()
        d=self.decision({n:copy.deepcopy(good)for n in policy.CANDIDATES})
        self.assertEqual(d['candidates'],list(policy.CANDIDATES));self.assertTrue(d['exceeds_budget_target'])

    def test_unknown_not_negative_and_no_elimination(self):
        _,good=fixtures();_,bad=fixtures(False)
        for r in bad['events']:r['generated']['visual']['nodding']['state']='unknown'
        d=self.decision(dict(baseline=good,event100=bad))
        self.assertEqual(len(d['candidates']),2);self.assertTrue(d['issues'])

    def test_quality_regression_blocks_event_only_win(self):
        _,good=fixtures();_,bad=fixtures(False)
        for r in good['video']:r['metrics']['lpips']=.8
        d=self.decision(dict(baseline=bad,event100=good))
        self.assertEqual(len(d['candidates']),2);self.assertFalse(d['dominance_current'])

    def test_baseline_kept_and_later_stages_require_repeat(self):
        _,good=fixtures();_,bad=fixtures(False)
        d=self.decision(dict(baseline=bad,event100=good));self.assertIn('baseline',d['candidates'])
        d=self.decision(dict(baseline=good,event100=bad),1500)
        self.assertEqual(len(d['candidates']),2)
        d=self.decision(dict(baseline=good,event100=bad),1500,dict(dominance_current=[['baseline','event100']]))
        self.assertEqual(d['candidates'],['baseline'])

    def test_changed_reference_or_membership_rejected(self):
        _,good=fixtures();_,bad=fixtures(False)
        bad['events'][0]['reference_identity']='different'
        with self.assertRaises(ValueError):self.decision(dict(baseline=good,event100=bad))
        _,bad=fixtures(False);bad['audio'].pop()
        with self.assertRaises(ValueError):self.decision(dict(baseline=good,event100=bad))


if __name__=='__main__':unittest.main(verbosity=2)
