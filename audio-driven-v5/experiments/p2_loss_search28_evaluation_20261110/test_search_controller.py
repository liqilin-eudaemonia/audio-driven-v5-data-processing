import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import promotion_policy
import search_controller as controller
import stage_driver
import p2_loss_search28.queue as training_queue


class SearchControllerChecks(unittest.TestCase):
    def setup_files(self,root):
        work=root/'evaluation';train=root/'training';work.mkdir();train.mkdir()
        (work/'PROMOTION_POLICY.json').write_text(json.dumps(promotion_policy.DEFAULT))
        (train/'PREFLIGHT_ACCEPTED.json').write_text(json.dumps(dict(source={})))
        (work/'AUTOMATION_CPU_ACCEPTED.json').write_text(json.dumps(dict(success=True,source={})))
        (work/'STAGE_500_STATUS.json').write_text(json.dumps(dict(stage='generation_review_ready')))
        return work,train

    def test_actual_sequence_requires_score_before_each_promotion(self):
        actions=[]
        def promote(step):
            actions.append(('promote',step))
            return dict(candidates=['baseline'],exceeds_budget_target=False,extra_update_steps_vs_target=0)
        with tempfile.TemporaryDirectory()as folder:
            work,train=self.setup_files(Path(folder))
            with patch.object(controller,'WORK',work),patch.object(controller,'TRAIN',train),\
                 patch.object(controller,'source_identity',return_value={}),\
                 patch.object(promotion_policy,'promote',side_effect=promote),\
                 patch.object(training_queue,'run',side_effect=lambda:actions.append(('train',))),\
                 patch.object(stage_driver,'evaluate',side_effect=lambda target:actions.append(('evaluate',target))),\
                 patch.object(stage_driver,'stage_candidates',return_value=['baseline']),\
                 patch.object(promotion_policy,'load_stage',return_value=({},[],{'evidence':'hash'})):
                controller.run()
            self.assertEqual(actions,[('promote',500),('train',),('evaluate',1500),
                ('promote',1500),('train',),('evaluate',3000),('promote',3000),('train',),('evaluate',6000)])
            result=json.loads((work/'FINALISTS_READY.json').read_text())
            self.assertFalse(result['best_model_selected']);self.assertTrue(result['pending'])

    def test_training_stop_never_launches_next_stage(self):
        with tempfile.TemporaryDirectory()as folder:
            work,train=self.setup_files(Path(folder));(train/'STOP').touch()
            with patch.object(controller,'WORK',work),patch.object(controller,'TRAIN',train),\
                 patch.object(controller,'source_identity',return_value={}),\
                 patch.object(training_queue,'run')as launch:
                controller.run();launch.assert_not_called()

    def test_changed_reviewed_code_prevents_launch(self):
        with tempfile.TemporaryDirectory()as folder:
            work,train=self.setup_files(Path(folder));source=work/'owned.py';source.write_text('changed')
            (work/'AUTOMATION_CPU_ACCEPTED.json').write_text(json.dumps(dict(success=True,source={str(source):'oldhash'})))
            with patch.object(controller,'WORK',work),patch.object(controller,'TRAIN',train),\
                 patch.object(controller,'source_identity',return_value={}):
                with self.assertRaises(ValueError):controller.run()


if __name__=='__main__':unittest.main(verbosity=2)
