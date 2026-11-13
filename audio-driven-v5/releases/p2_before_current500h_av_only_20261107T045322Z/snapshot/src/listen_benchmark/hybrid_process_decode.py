"""Opt-in spawned hybrid decoders; transfer only trusted cache paths over IPC."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

from .process_decode import ProcessMediaDecoder


def validate_device_mask(mask):
    if not isinstance(mask, str) or mask not in {str(i) for i in range(8)}:
        raise ValueError('Hybrid decoder requires exactly one explicit node17 GPU')


def init_hybrid_decoder(mask):
    validate_device_mask(mask)
    os.environ.update(CUDA_VISIBLE_DEVICES=mask, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def decode_hybrid_to_cache(task):
    row, directory, resources, slots, fast_crop, mask = task
    from .hybrid_video_decode import prepare_hybrid_media
    from .visual_media_cache import cached_media, cache_file
    from .visual_media_coverage import validate_terminal_support, VALIDATOR_VERSION
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from listen_visual_native8_nod import decode_slot
    reply = {'sample_id': row['sample_id'], 'decoder_pid': os.getpid(),
             'decoder_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        validate_device_mask(mask)
        if reply['decoder_cuda_visible_devices'] != mask:
            raise RuntimeError('Hybrid decoder GPU mask mismatch')

        def raw():
            started = time.monotonic()
            with decode_slot(Path(resources), slots):
                acquired = time.monotonic()
                media, info = prepare_hybrid_media(row['source'], device='0', fast_crop=fast_crop)
            return media, {'decode_s': time.monotonic() - acquired,
                           'decode_lock_wait_s': acquired - started, 'hybrid_decode': info}

        media, metrics = cached_media(row['source'], directory, raw)
        video = media[8]['meta']['video']
        validate_terminal_support(video['decoded_end_s'], media[8]['duration_s'],
                                  video['source_fps'], len(media[8]['images']))
        metrics = dict(metrics, input_coverage_validator=VALIDATOR_VERSION)
        if metrics.get('media_cache_hit'):
            metrics['hybrid_decode'] = {'reused_exact_media_cache': True, 'cuda_used': False}
        reply.update(cache_path=str(cache_file(row['source'], directory)), metrics=metrics)
    except Exception as exc:
        reply['error'] = repr(exc)
    return reply


def read_hybrid_reply(row, reply, directory, mask):
    from .visual_media_cache import cache_file
    from .j_annotation import locked
    validate_device_mask(mask)
    if reply.get('sample_id') != row['sample_id'] or reply.get('decoder_cuda_visible_devices') != mask:
        raise ValueError('Hybrid decoder sample/device mismatch')
    if 'error' in reply:
        return row, None, {'error': reply['error']}
    expected = cache_file(row['source'], directory)
    if reply.get('cache_path') != str(expected):
        raise ValueError('Hybrid decoder cache path mismatch')
    started = time.monotonic()
    with locked(expected.with_suffix('.lock')):
        with expected.open('rb') as f:
            media = pickle.load(f)
    metrics = dict(reply['metrics'], decode_backend='hybrid', decoder_pid=reply['decoder_pid'],
                   decoder_cuda_visible_devices=mask, cache_transfer_s=time.monotonic() - started)
    return row, media, metrics


class HybridMediaDecoder(ProcessMediaDecoder):
    def __init__(self, directory, resources, fast_crop, mask):
        validate_device_mask(mask)
        super().__init__(directory, resources, fast_crop)
        self.mask = mask

    def map(self, rows, workers, slots):
        if self.closed:
            raise RuntimeError('Hybrid decoder is closed')
        if type(workers) is not int or workers < 1 or type(slots) is not int or slots < 1:
            raise ValueError('Invalid hybrid decoding capacity')
        if self.pool is None or workers != self.workers:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                            initializer=init_hybrid_decoder, initargs=(self.mask,))
            self.workers = workers
        rows = list(rows)
        tasks = [(row, self.directory, self.resources, slots, self.fast_crop, self.mask) for row in rows]
        for row, reply in zip(rows, self.pool.map(decode_hybrid_to_cache, tasks)):
            yield read_hybrid_reply(row, reply, self.directory, self.mask)
