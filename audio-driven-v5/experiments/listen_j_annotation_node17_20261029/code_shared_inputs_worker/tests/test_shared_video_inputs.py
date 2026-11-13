"""Check optional shared arrays against the unchanged per-window input protocol."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.full_raw import views
from listen_benchmark.shared_video_inputs import prepare_shared_inputs
from listen_benchmark.visual_native8 import prepare_input


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps([messages, kwargs], sort_keys=True)


def media(duration):
    count = math.ceil(duration * 8)
    images = [Image.fromarray(np.full((512, 512, 3), i % 256, dtype=np.uint8))
              for i in range(count)]
    records = [{'time_s': i / 8, 'source_time_s': 10.123 + i / 8,
                'source_pts': i * 5, 'jpeg_sha256': str(i)} for i in range(count)]
    return {8: {'images': images, 'audio': None, 'duration_s': duration,
                'meta': {'video': {'frames': records, 'source_fps': 25,
                                   'last_frame_s': records[-1]['time_s'],
                                   'decoded_end_s': duration}}}}


class SharedInputTests(unittest.TestCase):
    def test_all_window_pixels_prompts_and_evidence_match(self):
        processor = Processor()
        for duration in (0.125, 2.99, 3.04, 10.37):
            with self.subTest(duration=duration):
                whole = media(duration)
                original_meta = json.dumps(whole[8]['meta'], sort_keys=True)
                shared = prepare_shared_inputs(processor, whole)
                full_frames = shared['full']['input'][0]['multi_modal_data']['video'][0][0]
                for view, start, end in views(duration):
                    old, evidence = prepare_input(processor, whole, start, end, view)
                    new, new_evidence = shared[view]['input']
                    self.assertEqual(evidence, new_evidence)
                    self.assertEqual(old['prompt'], new['prompt'])
                    self.assertEqual(old['mm_processor_kwargs'], new['mm_processor_kwargs'])
                    old_array, old_meta = old['multi_modal_data']['video'][0]
                    new_array, new_meta = new['multi_modal_data']['video'][0]
                    self.assertEqual(old_meta, new_meta)
                    np.testing.assert_array_equal(old_array, new_array)
                    self.assertTrue(new_array.flags.c_contiguous)
                    self.assertTrue(np.shares_memory(full_frames, new_array))
                self.assertEqual(original_meta, json.dumps(whole[8]['meta'], sort_keys=True))

    def test_bad_arrays_rejected(self):
        whole = media(1)
        for array in (np.zeros((7, 512, 512, 3), dtype=np.uint8),
                      np.zeros((8, 512, 512, 3), dtype=np.float32),
                      np.zeros((8, 512, 512, 3), dtype=np.uint8)[:, :, ::-1, :]):
            with self.assertRaisesRegex(ValueError, 'Preassembled native frames'):
                prepare_input(Processor(), whole, 0., 1., 'full', frame_array=array)


if __name__ == '__main__':
    unittest.main()
