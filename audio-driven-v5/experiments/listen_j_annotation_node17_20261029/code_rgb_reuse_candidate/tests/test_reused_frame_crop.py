from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
import numpy as np
try:
    import av
except ImportError:
    av = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark import frame_crop as crop


@unittest.skipIf(av is None, 'PyAV required for real pixel parity')
class ReusedCropTests(unittest.TestCase):
    def frame(self, width, height, fmt, seed, colorspace=2, color_range=0):
        f = av.VideoFrame(width, height, fmt)
        rng = np.random.default_rng(seed)
        dtype = np.dtype('<u2' if fmt == 'yuv420p10le' else 'uint8')
        upper = 1024 if dtype.itemsize == 2 else 256
        for p in f.planes:
            data = rng.integers(0, upper, size=(p.height, p.line_size // dtype.itemsize), dtype=dtype)
            p.update(data.tobytes())
        f.colorspace, f.color_range = colorspace, color_range
        return f

    def transform(self, frame, x=3, y=5):
        return {'source_size': [frame.width, frame.height], 'output_size': [512, 512], 'resample': 'bicubic',
                'crop': {'x': x, 'y': y, 'width': frame.width - x - 3, 'height': frame.height - y - 1}}

    def test_switching_sizes_formats_and_color_matrices_keeps_exact_pixels(self):
        specs = [(64, 80, 'yuv420p', 2, 0), (96, 128, 'yuv420p10le', 1, 1),
                 (64, 80, 'yuv420p', 1, 2), (96, 128, 'yuv420p10le', 2, 0),
                 (80, 96, 'rgb24', 2, 0), (64, 80, 'yuv420p', 2, 0)]
        for seed, (w, h, fmt, cs, cr) in enumerate(specs):
            with self.subTest(size=(w, h), fmt=fmt, colorspace=cs, range=cr):
                f = self.frame(w, h, fmt, seed, cs, cr)
                transform = self.transform(f)
                expected = crop.crop_original_frame(f, transform).tobytes()
                self.assertEqual(crop.crop_original_frame_reused(f, transform).tobytes(), expected)

    def test_thread_local_converters_do_not_share_mutable_swscale_state(self):
        barrier = threading.Barrier(2)

        def run(seed):
            f = self.frame(96, 128, ['yuv420p', 'yuv420p10le'][seed], seed)
            transform = self.transform(f)
            expected = crop.crop_original_frame(f, transform).tobytes()
            barrier.wait()
            for _ in range(3):
                self.assertEqual(crop.crop_original_frame_reused(f, transform).tobytes(), expected)
            return id(crop._conversion_state.converter)

        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(run, range(2)))
        self.assertEqual(len(set(ids)), 2)

    def test_invalid_crop_and_wrong_source_size_still_fail(self):
        f = self.frame(64, 80, 'yuv420p', 0)
        bad = self.transform(f)
        bad['source_size'] = [80, 96]
        with self.assertRaisesRegex(ValueError, 'source size mismatch'):
            crop.crop_original_frame_reused(f, bad)
        with self.assertRaises(ValueError):
            crop.crop_original_frame_reused(f, self.transform(f, x=-1))


if __name__ == '__main__':
    unittest.main()
