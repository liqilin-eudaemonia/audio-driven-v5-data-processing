"""Numerical checks independent of extraction, with known analytic answers."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("motion", Path(__file__).resolve().parents[1]/"scripts/p2_vico_motion_metrics.py")
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MotionMetricsTests(unittest.TestCase):
    def test_frechet_shift_and_scale_analytic(self):
        x = np.array([[-1.], [0.], [1.]])
        self.assertAlmostEqual(m.frechet(x, x+2), 4)
        self.assertAlmostEqual(m.frechet(x, 2*x), 1)
        self.assertEqual(m.frechet(x, x), 0)

    def test_frechet_rank_deficient_symmetric(self):
        r = np.random.default_rng(3)
        x, y = r.normal(size=(20, 64)), r.normal(size=(25, 64))
        self.assertAlmostEqual(m.frechet(x, y), m.frechet(y, x), places=5)

    def test_tlcc_known_delay(self):
        s = np.random.default_rng(7).normal(size=(200, 1))
        listener = np.concatenate((np.zeros((4, 1)), s[:-4]))
        curve, _ = m.tlcc(s, listener, 10)
        self.assertEqual(int(curve.argmax())-10, 4)
        self.assertAlmostEqual(curve[14, 0], 1)

    def test_constant_and_analytic_fd_variance(self):
        t = np.arange(180, dtype=float)[:, None]
        gt = np.tile(t, (1, 70));pred = gt+2
        score, audit = m.score_clip(gt, gt, pred)
        self.assertEqual(score["FD"]["exp"], 2)
        self.assertAlmostEqual(score["RTLCC"]["pose"], 0)
        self.assertAlmostEqual(score["FID_delta_fm"]["pose"], 0)
        self.assertAlmostEqual(score["V_D"]["exp"], (180**2-1)/12)
        self.assertEqual(audit["window_starts_frames"], [0, 30, 60])
        curve, undefined = m.tlcc(np.ones((180, 2)), np.ones((180, 2)))
        self.assertTrue((curve == 0).all());self.assertEqual(undefined, 121*2)

    def test_gt_identity_and_snd_sum(self):
        r = np.random.default_rng(11)
        s, g, p = [r.normal(size=(161, 70)) for _ in range(3)]
        score, _ = m.score_clip(s, g, g)
        for key in ("FD", "RTLCC", "RWTLCC", "FID_fm", "FID_delta_fm", "SND"):
            self.assertTrue(all(x == 0 for x in score[key].values()))
        score, audit = m.score_clip(s, g, p)
        self.assertEqual(audit["window_uncovered_tail_frames"], 11)
        for group in ("exp", "pose"):
            self.assertAlmostEqual(score["SND"][group], score["FID_fm"][group]+score["FID_delta_fm"][group])

    def test_misalignment_rejected(self):
        x = np.zeros((150, 70))
        with self.assertRaises(ValueError):m.score_clip(x, x, x[:-1])


if __name__ == "__main__":unittest.main()
