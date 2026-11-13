from pathlib import Path
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.hybrid_process_decode import read_hybrid_reply, init_hybrid_decoder, HybridMediaDecoder
from listen_benchmark.visual_media_cache import cache_file


class HybridProcessTests(unittest.TestCase):
    def test_cache_reply_preserves_pixels_and_rejects_other_gpu_or_other_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive'
            archive.write_bytes(b'original')
            row = {'sample_id': 'clip', 'source': {'source': {'archive': str(archive)}}}
            path = cache_file(row['source'], root)
            media = {8: {'images': [b'original-frame'], 'meta': {'pts': [1, 2]}}}
            path.write_bytes(pickle.dumps(media))
            reply = {'sample_id': 'clip', 'cache_path': str(path), 'decoder_cuda_visible_devices': '6',
                     'decoder_pid': 123, 'metrics': {'hybrid_decode': {'cuda_used': True}}}
            result = read_hybrid_reply(row, reply, root, '6')
            self.assertEqual(result[:2], (row, media))
            self.assertEqual(result[2]['decode_backend'], 'hybrid')
            self.assertEqual(result[2]['decoder_cuda_visible_devices'], '6')
            for replacement in [{'sample_id': 'another'}, {'decoder_cuda_visible_devices': '7'},
                                {'decoder_cuda_visible_devices': ''}, {'cache_path': '/untrusted'}]:
                with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                    read_hybrid_reply(row, dict(reply, **replacement), root, '6')

    def test_initializer_cannot_expose_multiple_or_implicit_gpus(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
            for mask in ['', '0,1', '-1', 'all', None]:
                with self.subTest(mask=mask), self.assertRaises(ValueError):
                    init_hybrid_decoder(mask)
            init_hybrid_decoder('6')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '6')
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')

    def test_spawn_capacity_change_drains_pool_and_keeps_single_gpu_assignment(self):
        instances = []

        class Pool:
            def __init__(self, **kwargs):
                self.kwargs, self.closed = kwargs, False
                instances.append(self)

            def map(self, fn, tasks):
                return [{'sample_id': t[0]['sample_id'], 'decoder_cuda_visible_devices': t[-1], 'error': 'test'}
                        for t in tasks]

            def shutdown(self, **kwargs):
                self.closed = True

        with patch('listen_benchmark.hybrid_process_decode.ProcessPoolExecutor', Pool):
            decoder = HybridMediaDecoder('cache', 'slots', True, '6')
            for n in [2, 2, 4]:
                result = list(decoder.map(iter([{'sample_id': 'clip'}]), n, 112))
                self.assertEqual(result, [({'sample_id': 'clip'}, None, {'error': 'test'})])
            self.assertEqual(len(instances), 2)
            self.assertTrue(instances[0].closed)
            self.assertEqual(instances[1].kwargs['initargs'], ('6',))
            self.assertEqual(instances[1].kwargs['mp_context'].get_start_method(), 'spawn')
            decoder.close()
            self.assertTrue(instances[1].closed)


if __name__ == '__main__':
    unittest.main()
