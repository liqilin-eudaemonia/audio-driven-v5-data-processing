"""Spawn-only CPU media decoding through the existing trusted frame cache."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time


def init_cpu_decoder():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def decode_to_cache(task):
    row, directory, resources, slots, fast_crop = task
    from .visual_native8 import prepare_media
    from .frame_crop import crop_original_frame
    from .visual_media_cache import cached_media, cache_file
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from listen_visual_native8_nod import decode_slot
    reply={'sample_id':row['sample_id'],'decoder_pid':os.getpid(),
           'decoder_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        if reply['decoder_cuda_visible_devices']!='':
            raise RuntimeError('CPU decoder must not expose CUDA devices')
        def raw():
            started=time.monotonic()
            with decode_slot(Path(resources),slots):
                acquired=time.monotonic()
                media=prepare_media(row['source'],frame_crop=crop_original_frame if fast_crop else None)
            return media,{'decode_s':time.monotonic()-acquired,'decode_lock_wait_s':acquired-started}
        media,metrics=cached_media(row['source'],directory,raw)
        reply.update(cache_path=str(cache_file(row['source'],directory)),metrics=metrics)
        del media
    except Exception as exc:
        reply['error']=repr(exc)
    return reply


def read_decoded_reply(row, reply, directory):
    from .visual_media_cache import cache_file
    from .j_annotation import locked
    if reply.get('sample_id')!=row['sample_id'] or reply.get('decoder_cuda_visible_devices')!='':
        raise ValueError('CPU decoder identity/device mismatch')
    if 'error' in reply:return row,None,{'error':reply['error']}
    expected=cache_file(row['source'],directory)
    if reply.get('cache_path')!=str(expected):raise ValueError('CPU decoder cache path mismatch')
    started=time.monotonic()
    with locked(expected.with_suffix('.lock')):
        with expected.open('rb') as f:media=pickle.load(f)
    metrics=dict(reply['metrics'],decode_backend='process',decoder_pid=reply['decoder_pid'],
                 decoder_cuda_visible_devices='',cache_transfer_s=time.monotonic()-started)
    return row,media,metrics


class ProcessMediaDecoder:
    def __init__(self, directory, resources, fast_crop):
        self.directory=str(directory);self.resources=str(resources);self.fast_crop=fast_crop
        self.pool=None;self.workers=None;self.closed=False

    def map(self, rows, workers, slots):
        if self.closed:raise RuntimeError('CPU decoder is closed')
        if type(workers) is not int or workers<1 or type(slots) is not int or slots<1:
            raise ValueError('Invalid CPU decoding capacity')
        if self.pool is None or workers!=self.workers:
            if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True)
            self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                          initializer=init_cpu_decoder)
            self.workers=workers
        tasks=[(row,self.directory,self.resources,slots,self.fast_crop) for row in rows]
        for row,reply in zip(rows,self.pool.map(decode_to_cache,tasks)):
            yield read_decoded_reply(row,reply,self.directory)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True)
