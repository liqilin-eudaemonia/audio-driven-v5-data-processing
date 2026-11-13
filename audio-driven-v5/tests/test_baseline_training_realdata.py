"""Regression checks for actual source binding and deterministic native kernels."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from baseline_training.data import WORK, SelectedPool, read, sha
from baseline_training.engine import check_features
from baseline_training.ops import token_cross_entropy, linear_audio_interpolate


class ActualDataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read(WORK / 'DATA_PROTOCOL.json')
        cls.ids = cls.protocol['diagnostic_train'] + cls.protocol['diagnostic_validation']
        cls.pool = SelectedPool()

    def test_exact_final_pool_caption_crop_and_disjoint_groups(self):
        self.assertEqual(len(self.pool.ids), 284218)
        groups = self.protocol['diagnostic_groups']
        self.assertFalse({groups[x] for x in self.protocol['diagnostic_train']} &
                         {groups[x] for x in self.protocol['diagnostic_validation']})
        for sid in self.ids:
            row, item = self.pool.row(sid)
            media = read(WORK / 'media' / sid / 'receipt.json')
            self.assertEqual(media['contract']['caption'], row['caption'])
            self.assertEqual(media['contract']['caption_sha256'], row['caption_file_sha256'])
            self.assertEqual(media['contract']['listener_source'], row['source'])
            self.assertIn(row['split'], ['train', 'dev'])
            self.assertFalse(media['future_listener_input'])
            for name, expected in media['files'].items():
                self.assertEqual(sha(WORK / 'media' / sid / name), expected)

    def test_native_spaces_and_all_real_detector_support(self):
        for sid in self.ids:
            for method, width in [('bfm', 73), ('l2l', 56), ('dim', 56), ('dystream', 512), ('avatarforcing', 512)]:
                value = check_features(WORK, method, sid)
                self.assertEqual(value['listener'].shape[-1], width)
                self.assertTrue(torch.isfinite(value['listener']).all())
                if method == 'bfm':
                    self.assertTrue(value['listener_detected'].all())
                    self.assertTrue(value['speaker_detected'].all())
                if method == 'dim':
                    self.assertEqual(value['audio'].shape[-1], 768)

    def test_unknown_reactions_are_masked(self):
        for sid in self.ids:
            value = check_features(WORK, 'glare', sid)
            mask = value['reaction_valid']
            self.assertEqual(mask.shape, value['reaction'].shape)
            self.assertEqual(mask.dtype, torch.bool)
            quality = read(WORK / 'reaction' / sid / 'quality.json')
            if quality['fer_valid_frames'] != len(mask):
                self.assertFalse(mask[:, :4].any())
        self.assertTrue(any(not check_features(WORK, 'glare', sid)['reaction_valid'].all() for sid in self.ids))

    def test_changed_source_and_feature_are_rejected(self):
        sid = self.ids[0]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'features/bfm').mkdir(parents=True)
            (work / 'media' / sid).mkdir(parents=True)
            for ext in ['pt', 'json']:
                shutil.copyfile(WORK / 'features/bfm' / (sid + '.' + ext), work / 'features/bfm' / (sid + '.' + ext))
            source = work / 'media' / sid / 'receipt.json'
            shutil.copyfile(WORK / 'media' / sid / 'receipt.json', source)
            check_features(work, 'bfm', sid)
            source.write_text(source.read_text() + ' ')
            with self.assertRaisesRegex(AssertionError, 'Media changed'):
                check_features(work, 'bfm', sid)
            shutil.copyfile(WORK / 'media' / sid / 'receipt.json', source)
            with (work / 'features/bfm' / (sid + '.pt')).open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(AssertionError, 'Feature changed'):
                check_features(work, 'bfm', sid)

    def test_extra_speaker_rotation_matches_reviewed_pixels(self):
        from PIL import Image
        review = read(WORK / 'SPEAKER_ORIENTATION.json')['5088fb158abbbd45a5e05b04']
        self.assertEqual(sha(WORK / review['image']), review['image_sha256'])
        sid = self.protocol['diagnostic_validation'][0]
        image = np.array(Image.open(WORK / review['image']))
        actual = np.load(WORK / 'media' / sid / 'speaker.npy', mmap_mode='r')[0]
        np.testing.assert_array_equal(actual, np.rot90(image))


class EquivalentKernels(unittest.TestCase):
    def test_token_ce_matches_native_value_and_gradient(self):
        torch.manual_seed(137)
        logits = torch.randn(2, 17, 11, requires_grad=True)
        target = torch.randint(0, 17, (2, 11))
        target[0, 3] = -100
        original = F.cross_entropy(logits, target, ignore_index=-100)
        reference = torch.autograd.grad(original, logits)[0]
        changed = token_cross_entropy(logits, target, ignore_index=-100)
        actual = torch.autograd.grad(changed, logits)[0]
        torch.testing.assert_close(changed, original)
        torch.testing.assert_close(actual, reference)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for native-kernel equivalence check')
    def test_audio_interpolation_preserves_forward_and_trainable_gradient(self):
        torch.manual_seed(137)
        value = torch.randn(1, 768, 99, device='cuda', requires_grad=True)
        output = linear_audio_interpolate(value, scale_factor=.5, mode='linear', align_corners=True)
        with torch.no_grad():
            native = F.interpolate(value, scale_factor=.5, mode='linear', align_corners=True)
        torch.testing.assert_close(output, native, rtol=1e-5, atol=2e-6)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.square().sum()), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
