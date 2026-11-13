import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import eval_runtime
import stage_driver
import promotion_policy
from p2_listen.contracts import atomic_json,sha
from p2_event_presence.contracts import digest


class ResumeEvidenceChecks(unittest.TestCase):
    def scored_fixture(self,root):
        train=root/'train';work=root/'eval/evaluations/baseline_500';work.mkdir(parents=True)
        cp=train/'runs/baseline/step_000500.pt';cp.parent.mkdir(parents=True)
        atomic_json(cp.with_suffix('.complete.json'),dict(sha256='checkpoint-hash'))
        atomic_json(work/'REQUEST.json',dict(checkpoint=str(cp),samples=64))
        atomic_json(work/'protocol.json',dict(execution_code={'code':'hash'},total_samples=64,
            arms=[dict(name='arm',step=500,original_sha256='checkpoint-hash')],
            checkpoint_identity=dict(experiment=dict(loss_search=dict(candidate='baseline')))))
        ref=root/'eval/reference_events'/(digest('id')+'.json');atomic_json(ref,dict(label='unknown'))
        for f in ['FINAL_AUDIT.json','RESULTS.json','event_pairs_fixed_reference.jsonl','fixed_reference_event_summary.json',
                  'metrics/arm/seamless/pairs.jsonl','metrics/arm/seamless/audio_pairs.jsonl']:
            p=work/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
        atomic_json(work/'REFERENCE_EVENTS_ACCEPTED.json',dict(reference_sha256_by_id={'id':sha(ref)}))
        files=[f for f in work.rglob('*')if f.is_file()and f.name not in ('REQUEST.json','protocol.json')]
        atomic_json(work/'SCORING_COMPLETE.json',dict(success=True,checkpoint_sha256='checkpoint-hash',
            evidence={str(f.relative_to(work)):sha(f)for f in files}))
        return train,work

    def test_completed_score_validation_never_rewrites_evidence(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);_,work=self.scored_fixture(root);before={str(f):f.read_bytes()for f in work.rglob('*')if f.is_file()}
            with patch.object(stage_driver,'WORK',root/'eval'),patch.object(eval_runtime,'code_identity',return_value={'code':'hash'}),\
                 patch('p2_event_presence.media.verify')as media:
                stage_driver.verify_scoring(work);media.assert_called_once_with(work/'events')
            self.assertEqual(before,{str(f):f.read_bytes()for f in work.rglob('*')if f.is_file()})

    def test_corrupted_evidence_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);_,work=self.scored_fixture(root);(work/'RESULTS.json').write_text('{"changed":true}')
            with patch.object(stage_driver,'WORK',root/'eval'),patch.object(eval_runtime,'code_identity',return_value={'code':'hash'}):
                with self.assertRaises(ValueError):stage_driver.verify_scoring(work)

    def test_finished_stage_skips_generation_and_rebinding(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);train,work=self.scored_fixture(root)
            atomic_json(train/'runs/baseline/cluster.json',{})
            before=sha(work/'SCORING_COMPLETE.json')
            with patch.object(stage_driver,'WORK',root/'eval'),patch.object(stage_driver,'TRAIN',train),\
                 patch.object(stage_driver,'stage_candidates',return_value=['baseline']),\
                 patch.object(stage_driver,'checkpoint_receipts_ready',return_value=True),\
                 patch.object(stage_driver,'verify_scoring')as verify,\
                 patch.object(eval_runtime,'prepare',side_effect=AssertionError('must not regenerate')),\
                 patch.object(eval_runtime,'final',side_effect=AssertionError('must not rewrite final audit')):
                stage_driver.evaluate(500);verify.assert_called_once_with(work)
            self.assertEqual(sha(work/'SCORING_COMPLETE.json'),before)

    def test_promotion_resume_reuses_sealed_decision(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);train=root/'train';train.mkdir();work=root/'eval';work.mkdir()
            atomic_json(work/'PROMOTION_POLICY.json',promotion_policy.DEFAULT)
            d=dict(completed_step=500,target=1500,candidates=['baseline'])
            with patch.object(promotion_policy,'WORK',work),patch.object(promotion_policy,'TRAIN',train),\
                 patch.object(stage_driver,'stage_candidates',return_value=['baseline']),\
                 patch.object(promotion_policy,'load_stage',return_value=({},[],{'score':'sealed-hash'})),\
                 patch.object(promotion_policy,'decide',return_value=d)as decide:
                promotion_policy.promote(500);before=(work/'DECISION_500.json').read_bytes()
                promotion_policy.promote(500);self.assertEqual(decide.call_count,1)
                self.assertEqual((work/'DECISION_500.json').read_bytes(),before)
            with patch.object(promotion_policy,'WORK',work),patch.object(promotion_policy,'TRAIN',train),\
                 patch.object(stage_driver,'stage_candidates',return_value=['baseline']),\
                 patch.object(promotion_policy,'load_stage',return_value=({},[],{'score':'changed-hash'})):
                with self.assertRaises(ValueError):promotion_policy.promote(500)


if __name__=='__main__':unittest.main(verbosity=2)
