"""CPU checks for the non-prosody cache contract and resumable commit."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('glare_nonprosody', ROOT / 'scripts/prepare_glare_nonprosody_cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CacheContract(unittest.TestCase):
    def setUp(self):
        self.protocol = cache.load_protocol(cache.DEFAULT)
        db = sqlite3.connect((cache.DEFAULT / 'jobs.sqlite').as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        self.job = dict(db.execute('SELECT * FROM jobs LIMIT 1').fetchone())
        db.close()

    def test_selection_is_exact_and_shards_are_disjoint(self):
        db = sqlite3.connect((cache.DEFAULT / 'jobs.sqlite').as_uri() + '?mode=ro', uri=True)
        self.assertEqual(db.execute('SELECT COUNT(*),COUNT(DISTINCT sid) FROM jobs').fetchone(), (284218, 284218))
        self.assertEqual(dict(db.execute('SELECT split,COUNT(*) FROM jobs GROUP BY split')), {'train': 281060, 'dev': 3158})
        self.assertEqual([x[0] for x in db.execute('SELECT COUNT(*) FROM jobs GROUP BY worker ORDER BY worker')], self.protocol['worker_counts'])
        self.assertAlmostEqual(db.execute('SELECT SUM(duration)/3600 FROM jobs').fetchone()[0], self.protocol['hours'], places=7)
        db.close()

    def test_selected_row_and_caption_bound_evidence(self):
        row = cache.load_row(self.job, self.protocol)
        value = cache.evidence(row)
        self.assertIn(value['status'], ['indexed', 'source_differs_requires_review', 'missing_production_evidence'])
        self.assertFalse(value['training_labels_admitted'])

    def test_commit_reopen_and_corruption_rejection(self):
        n = __import__('math').ceil(self.job['duration'] * 25 - 1e-7)
        motion = torch.zeros(n, 512)
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(cache.existing(directory, self.job, self.protocol))
            cache.commit(directory, self.job, self.protocol, motion, [0.] * n,
                         {'status': 'missing_production_evidence', 'training_labels_admitted': False})
            self.assertTrue(cache.existing(directory, self.job, self.protocol))
            payload, receipt = cache.paths(directory, self.job['sid'])
            self.assertEqual(set(torch.load(payload, weights_only=True)), {'motion'})
            self.assertFalse(json.loads(receipt.read_text())['full_glare_training_admitted'])
            with payload.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(AssertionError, 'corrupt'):
                cache.existing(directory, self.job, self.protocol)

    def test_changed_source_identity_is_rejected(self):
        changed = dict(self.job, row_sha='bad')
        with self.assertRaisesRegex(AssertionError, 'row changed'):
            cache.load_row(changed, self.protocol)

    def test_no_prosody_or_label_admission_claim(self):
        self.assertEqual(self.protocol['prosody'], 'deferred_by_user')
        self.assertFalse(self.protocol['raw_media_export'])
        self.assertFalse(self.protocol['full_glare_training_admitted'])

    def test_unsafe_sample_path_rejected(self):
        with self.assertRaises(AssertionError):
            cache.paths('/tmp/unused', '../bad')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
