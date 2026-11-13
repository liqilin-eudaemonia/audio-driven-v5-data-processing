import tempfile
import unittest
from pathlib import Path
import numpy as np
from .cache import GTFeatureCache


class GTCacheTests(unittest.TestCase):
    def test_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = GTFeatureCache(folder, {'model_sha': 'a', 'device': 'cpu'})
            frames = np.zeros((2, 4, 4, 3), np.uint8)
            calls = []
            def compute():
                calls.append(1)
                return np.array([[np.nan, 1.]]), np.array([2], dtype=np.int64)
            original, hit = cache.get(frames, 'faces', {'padding': 64}, compute)
            self.assertFalse(hit)
            original[1][0] = 999  # Caller mutation cannot alter stored features.
            result, hit = cache.get(frames, 'faces', {'padding': 64}, compute)
            self.assertTrue(hit)
            self.assertEqual(result[1][0], 2)
            self.assertTrue(np.isnan(result[0][0, 0]))
            for changed, parameters in [(frames.copy(), {'padding': 0}),
                                          (np.ones_like(frames), {'padding': 64})]:
                _, hit = cache.get(changed, 'faces', parameters, compute)
                self.assertFalse(hit)
            other = GTFeatureCache(folder, {'model_sha': 'b', 'device': 'cpu'})
            _, hit = other.get(frames, 'faces', {'padding': 64}, compute)
            self.assertFalse(hit)
            self.assertEqual(len(calls), 4)

    def test_failed_compute_is_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = GTFeatureCache(folder, {})
            def fail():
                raise ValueError('feature extraction failed')
            with self.assertRaises(ValueError):
                cache.get(np.zeros((1, 2), np.uint8), 'fid', {}, fail)
            self.assertFalse(list(Path(folder).rglob('*.npz')))


if __name__ == '__main__':
    unittest.main()
