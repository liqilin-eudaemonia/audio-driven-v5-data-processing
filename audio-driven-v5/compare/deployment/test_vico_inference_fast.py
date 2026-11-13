"""Focused cache correctness and resume checks; no GPU or archived file writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vico_more_common import save, sha
from vico_inference_fast import verified_records


class CacheTests(unittest.TestCase):
    def test_corrupt_missing_and_duplicate_outputs_are_not_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'prediction.npz'
            data.write_bytes(b'original')
            r = dict(id='clip', path=str(data), sha256=sha(data))
            marker = root / 'COMPLETE.json'
            save(marker, dict(success=True, records=[r]))
            self.assertEqual(verified_records(marker, ['clip']), [r])
            self.assertIsNone(verified_records(marker, ['other']))
            save(marker, dict(success=True, records=[r, r]))
            self.assertIsNone(verified_records(marker, ['clip', 'other']))
            save(marker, dict(success=True, records=[r]))
            data.write_bytes(b'corrupted')
            self.assertIsNone(verified_records(marker, ['clip']))
            data.unlink()
            self.assertIsNone(verified_records(marker, ['clip']))

    def test_new_drivers_do_not_reuse_archived_render(self):
        from vico_inference_fast import reusable, WORK
        generation = json.loads((WORK / 'listenformer/GENERATION_COMPLETE.json').read_text())
        generation['records'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save(out / 'GENERATION_COMPLETE.json', generation)
            self.assertEqual(reusable('listenformer', 'render', out), (None, None))

    def test_lost_aggregate_manifest_resumes_without_gpu(self):
        from vico_inference_fast import generate_listenformer
        from vico_render_fast import render
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Existing official outputs are only read. GPU access is a test failure.
            with patch('vico_more_train.gpu_guard', side_effect=AssertionError('unexpected GPU request')):
                generate_listenformer(out)
            self.assertEqual(len(json.loads((out / 'GENERATION_COMPLETE.json').read_text())['records']), 43)
            with patch('vico_render_fast.gpu_guard', side_effect=AssertionError('unexpected GPU request')):
                render(out)
            self.assertEqual(len(json.loads((out / 'RENDER_COMPLETE.json').read_text())['records']), 43)


if __name__ == '__main__':
    unittest.main()
