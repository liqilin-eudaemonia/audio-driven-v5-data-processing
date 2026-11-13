from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark import hybrid_video_decode as hybrid


class Frame:
    def __init__(self, fmt):
        self.format = SimpleNamespace(name=fmt)
        self.conversions = []

    def reformat(self, *, format):
        self.conversions.append(format)
        return Frame(format)


class HybridDecodeTests(unittest.TestCase):
    def run_path(self, source_format, hardware_error=None, hardware_format='nv12', cpu_error=None):
        row = {'source': {'original': True}}
        visited = []

        @contextmanager
        def cpu(source):
            self.assertEqual(source, row['source'])
            visited.append('cpu-open')
            c = SimpleNamespace(name='h264', profile='test', format=SimpleNamespace(name=source_format))
            try:
                yield SimpleNamespace(streams=SimpleNamespace(video=[SimpleNamespace(codec_context=c)]),
                                      frame=Frame(source_format)), 'original-reader'
            finally:
                visited.append('cpu-close')

        @contextmanager
        def gpu(source, device):
            self.assertEqual(visited[-1], 'cpu-close')
            self.assertEqual(device, '0')
            visited.append('cuda-open')
            if hardware_error:
                raise hardware_error
            yield SimpleNamespace(frame=Frame(hardware_format)), 'original-reader'

        def build(actual_row, frame_crop=None, source_opener=None):
            self.assertIs(actual_row, row)
            if source_opener is None:
                visited.append('full-cpu-recovery')
                return {'complete_cpu_media': True}
            with source_opener(actual_row['source']) as (c, reader):
                if cpu_error:
                    raise cpu_error
                return {'rgb_input_format': frame_crop(c.frame, {}), 'reader': reader}

        with patch.object(hybrid, 'open_source', cpu), patch.object(hybrid, 'open_cuda_source', gpu), \
             patch.object(hybrid, 'prepare_media_covered', build), \
             patch('listen_benchmark.frame_crop.crop_original_frame', lambda f, _: f.format.name):
            media, info = hybrid.prepare_hybrid_media(row)
        return media, info, visited

    def test_high10_never_opens_cuda_and_keeps_original_cpu_frame(self):
        media, info, visited = self.run_path('yuv420p10le')
        self.assertEqual(media['rgb_input_format'], 'yuv420p10le')
        self.assertFalse(info['cuda_attempted'])
        self.assertNotIn('cuda-open', visited)
        self.assertFalse(info['cpu_fallback'])

    def test_supported_nv12_is_normalized_before_original_rgb_crop(self):
        media, info, visited = self.run_path('yuv420p')
        self.assertEqual(media['rgb_input_format'], 'yuv420p')
        self.assertEqual(media['reader'], 'original-reader')
        self.assertTrue(info['cuda_used'])
        self.assertEqual(visited, ['cpu-open', 'cpu-close', 'cuda-open'])

    def test_hardware_failure_discards_attempt_and_recovers_full_cpu_media(self):
        for failure in [RuntimeError('unsupported by driver'), OSError('decoder unavailable')]:
            with self.subTest(failure=failure):
                media, info, visited = self.run_path('yuv420p', hardware_error=failure)
                self.assertEqual(media, {'complete_cpu_media': True})
                self.assertTrue(info['cuda_attempted'])
                self.assertTrue(info['cpu_fallback'])
                self.assertFalse(info['cuda_used'])
                self.assertIn(str(failure), info['cuda_error'])
                self.assertEqual(visited[-1], 'full-cpu-recovery')

    def test_unvalidated_hardware_pixel_format_is_not_used(self):
        media, info, _ = self.run_path('yuv420p', hardware_format='p010le')
        self.assertEqual(media, {'complete_cpu_media': True})
        self.assertTrue(info['cpu_fallback'])
        self.assertIsNone(info['pixel_normalization'])

    def test_cpu_source_error_and_host_memory_failure_are_not_hidden(self):
        with self.assertRaisesRegex(ValueError, 'invalid PTS'):
            self.run_path('yuv420p10le', cpu_error=ValueError('invalid PTS'))
        with self.assertRaises(MemoryError):
            self.run_path('yuv420p', hardware_error=MemoryError())


if __name__ == '__main__':
    unittest.main()
