import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .core import emotion_scores, frechet, json_safe, pixel_scores, response_scores, validate_intervals
from .media import load_pair
from .run import read_manifest
from .backends import normalize_landmarks


class MetricTests(unittest.TestCase):
    def test_pixels_identical_and_constant_error(self):
        x = np.zeros((16,16,3),np.uint8)
        result = pixel_scores(x,x)
        self.assertTrue(np.isinf(result['psnr']))
        self.assertEqual(result['ssim'],1.)
        other = pixel_scores(x,x+10)
        self.assertAlmostEqual(other['psnr'],20*np.log10(25.5))
        self.assertLess(other['ssim'],1.)

    def test_frechet_analytic_and_symmetry(self):
        x = np.array([[0.,0.],[2.,0.]])
        self.assertAlmostEqual(frechet(x,x),0.)
        self.assertAlmostEqual(frechet(x,x+1),2.)
        self.assertAlmostEqual(frechet(x,2*x),3.)
        self.assertAlmostEqual(frechet(x,2*x,population=True),2.)
        rng = np.random.default_rng(9)
        a,b = rng.normal(size=(13,4)),rng.normal(size=(17,4))
        self.assertAlmostEqual(frechet(a,b),frechet(b,a),places=9)

    def test_frechet_rejects_too_few_and_invalid(self):
        with self.assertRaises(ValueError): frechet([[1,2]],[[1,2]])
        with self.assertRaises(ValueError): frechet([[1],[np.nan]],[[1],[2]])

    def test_emotion_common_support_and_diversity(self):
        result = emotion_scores([0,1,2,-1,6],[0,2,-1,5,6])
        self.assertEqual(result['valid_frames'],3)
        self.assertEqual(result['matches'],2)
        self.assertEqual(result['diversity_gt'],3)
        self.assertEqual(result['diversity_generated'],3)
        self.assertIsNone(emotion_scores([-1],[0])['accuracy'])

    def test_response_abs_before_phase_averaging(self):
        gt = np.zeros((8,106,2))
        pred = gt.copy()
        gt[:,:,0] = np.arange(8)[:,None]
        pred[:,:,0] = np.array([0,2,4,6,6,6,6,6])[:,None]
        result = response_scores(gt,pred,2,[[0,2],[2,4]])
        # GT motion is 1 in each phase; generated motion is 2 then 0.
        self.assertEqual(result['delta'],1)
        self.assertEqual(result['valid_phase_duration'],4)

    def test_response_no_bridging_and_duration_weight(self):
        g = np.zeros((6,106,2)); p = g.copy()
        p[:,:,0] = np.array([0,2,999,9,9,9])[:,None]
        p[2] = np.nan
        r = response_scores(g,p,1,[[0,2],[2,6]])
        self.assertAlmostEqual(r['delta'],4/6)
        self.assertEqual(r['phases'][1]['valid_steps'],2)
        self.assertEqual(r['phases'][1]['possible_steps'],3)

    def test_intervals_invalid_overlap(self):
        with self.assertRaises(ValueError): validate_intervals([[1,3],[2,4]],4)
        with self.assertRaises(ValueError): validate_intervals([[0,5]],4)
        with self.assertRaises(ValueError): validate_intervals([[2,1]],4)

    def test_normalization_preserves_holes(self):
        lm = np.ones((3,106,2)); box = np.tile([0.,0.,224.,224.],(3,1))
        box[1] = np.nan; lm[1] = np.nan
        a,b,side = normalize_landmarks(lm,lm,box,box)
        self.assertEqual(side,224)
        self.assertTrue(np.isnan(a[1]).all())
        np.testing.assert_allclose(a,b)

    def test_json_inf_is_standard_json(self):
        text = json.dumps(json_safe({'psnr':float('inf')}),allow_nan=False)
        self.assertEqual(json.loads(text)['psnr'],'inf')
        with self.assertRaises(ValueError): json_safe(float('nan'))

    def test_manifest_duplicate_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'pairs.json'
            row = dict(id='a',gt='gt.mp4',generated='gen.mp4')
            path.write_text(json.dumps([row]))
            self.assertEqual(read_manifest(path)[0]['gt'],str(Path(d)/'gt.mp4'))
            path.write_text(json.dumps([row,row]))
            with self.assertRaises(ValueError): read_manifest(path)

    def test_media_trim_and_fps_rejection(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            def video(name,n,fps):
                path = str(Path(d)/name)
                writer = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(16,16))
                self.assertTrue(writer.isOpened())
                for i in range(n): writer.write(np.full((16,16,3),i*20,np.uint8))
                writer.release()
                return path
            gt = video('gt.mp4',4,25); pred = video('pred.mp4',3,25)
            a,b,meta = load_pair(dict(gt=gt,generated=pred))
            self.assertEqual(len(a),3); self.assertEqual(meta['trimmed_gt'],1)
            mismatch = video('fps.mp4',4,30)
            with self.assertRaises(ValueError): load_pair(dict(gt=gt,generated=mismatch))
            with self.assertRaises(ValueError): load_pair(dict(gt=gt,generated=pred,gt_roi=[0,0,100,100]))


if __name__ == '__main__':
    unittest.main()
