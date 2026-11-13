import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import stage_driver as driver


class StageDriverChecks(unittest.TestCase):
    def test_never_start_when_any_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, 'TRAIN', Path(folder)):
            out = Path(folder) / 'runs/baseline'; out.mkdir(parents=True)
            self.assertFalse(driver.checkpoint_receipts_ready(500, ['baseline']))
            (out / 'STAGE_500.json').write_text(json.dumps(dict(success=True, step=500, sha256='hash')))
            self.assertFalse(driver.checkpoint_receipts_ready(500, ['baseline']))
            (out / 'step_000500.complete.json').write_text(json.dumps(dict(sha256='hash')))
            self.assertTrue(driver.checkpoint_receipts_ready(500, ['baseline']))
            self.assertFalse(driver.checkpoint_receipts_ready(500, ['baseline', 'event001']))

    def test_mismatched_training_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, 'TRAIN', Path(folder)):
            out = Path(folder) / 'runs/baseline'; out.mkdir(parents=True)
            (out / 'STAGE_500.json').write_text(json.dumps(dict(success=True, step=500, sha256='old')))
            (out / 'step_000500.complete.json').write_text(json.dumps(dict(sha256='new')))
            with self.assertRaises(ValueError): driver.checkpoint_receipts_ready(500, ['baseline'])

    def test_later_stage_requires_matched_control(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, 'TRAIN', Path(folder)):
            f = Path(folder) / 'promotion_1500.json'
            f.write_text(json.dumps(dict(target=1500, basis='generated_av_quality', candidates=['event001'])))
            with self.assertRaises(ValueError): driver.stage_candidates(1500)
            f.write_text(json.dumps(dict(target=1500, basis='generated_av_quality', candidates=['event001', 'baseline'])))
            self.assertEqual(driver.stage_candidates(1500), ['baseline', 'event001'])


if __name__ == '__main__': unittest.main(verbosity=2)
