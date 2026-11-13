import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.shared_raw_decode import RawWriter
from listen_benchmark.shared_visual_decode import load_native
from listen_benchmark.frame_crop import crop_original_frame


class RawHandoffTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(video_transform=dict(source_size=[642, 480],
            crop=dict(x=17, y=23, width=333, height=333), output_size=[512, 512], resample='bicubic'))
        rgb = np.random.default_rng(9).integers(0, 256, (480, 642, 3), dtype=np.uint8)
        self.frame = av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(format='yuv420p10le')

    def test_exact_plane_reconstruction_with_padding_and_color_metadata(self):
        for space in [1, 2, 5]:
            for color_range in [1, 2]:
                self.frame.colorspace = space
                self.frame.color_range = color_range
                with self.subTest(space=space, range=color_range), tempfile.TemporaryDirectory() as d:
                    writer = RawWriter(d, self.source)
                    writer.append(self.frame, .02)
                    writer.append(self.frame, .06)
                    self.assertTrue(writer.finish('original'))
                    images, times, reader = load_native(d, self.source)
                    np.testing.assert_array_equal(images[0], np.asarray(crop_original_frame(self.frame, self.source['video_transform'])))
                    self.assertEqual(times, [.02, .06])
                    self.assertEqual(reader, 'original')
                    with self.assertRaises(ValueError):
                        load_native(d, {})

    def test_memory_budget_falls_back_without_publishing_manifest(self):
        with tempfile.TemporaryDirectory() as d, patch('listen_benchmark.shared_raw_decode.MAX_BYTES', 10):
            writer = RawWriter(d, self.source)
            writer.append(self.frame, 0.)
            self.assertFalse(writer.finish('original'))
            self.assertFalse((Path(d) / 'manifest.json').exists())

    def test_corrupt_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            writer = RawWriter(d, self.source)
            writer.append(self.frame, 0.)
            writer.finish('original')
            with (Path(d) / 'planes.bin').open('r+b') as f:
                f.write(b'BAD')
            with self.assertRaises(ValueError):
                load_native(d, self.source)


if __name__ == '__main__':
    unittest.main()
