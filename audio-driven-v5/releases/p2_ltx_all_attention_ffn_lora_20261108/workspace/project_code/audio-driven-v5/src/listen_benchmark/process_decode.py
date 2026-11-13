"""Spawn-only CPU media decoding through the existing trusted frame cache."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
from pathlib import Path
import pickle
import sys
import time


def init_cpu_decoder():
    owner=os.environ.get('CUDA_VISIBLE_DEVICES','')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      LISTEN_CPU_DECODER_OWNER_GPU=owner if owner in {str(i) for i in range(8)} else '')


def use_reused_rgb(resources, owner):
    """Hot opt-in by owning GPU; older settings retain the original RGB path."""
    path=Path(resources).parent/'resource_tuning.json'
    if not path.exists():return False
    gpus=json.loads(path.read_text()).get('rgb_reuse_gpus',[])
    if not isinstance(gpus,list) or any(type(x) is not int or not 0<=x<8 for x in gpus):
        raise ValueError('rgb_reuse_gpus must be a list of node17 GPU indices')
    return owner in {str(i) for i in gpus}


def decode_to_cache(task):
    row, directory, resources, slots, fast_crop = task
    from .visual_media_coverage import prepare_media_covered, validate_terminal_support, VALIDATOR_VERSION
    from .frame_crop import crop_original_frame, crop_original_frame_reused, reused_conversion_info
    from .visual_media_cache import cached_media, cache_file
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from listen_visual_native8_nod import decode_slot
    reply={'sample_id':row['sample_id'],'decoder_pid':os.getpid(),
           'decoder_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    owner=os.environ.get('LISTEN_CPU_DECODER_OWNER_GPU','')
    try:
        if reply['decoder_cuda_visible_devices']!='':
            raise RuntimeError('CPU decoder must not expose CUDA devices')
        reuse=fast_crop and use_reused_rgb(resources,owner)
        crop=crop_original_frame_reused if reuse else crop_original_frame
        def raw():
            started=time.monotonic()
            with decode_slot(Path(resources),slots):
                acquired=time.monotonic()
                media=prepare_media_covered(row['source'],frame_crop=crop if fast_crop else None)
            return media,{'decode_s':time.monotonic()-acquired,'decode_lock_wait_s':acquired-started,
                          'rgb_reuse_requested':bool(reuse),
                          'input_rgb_converter':reused_conversion_info() if reuse else {'path':'legacy_full_rgb' if fast_crop else 'legacy_pil'}}
        media,metrics=cached_media(row['source'],directory,raw)
        video=media[8]['meta']['video']
        validate_terminal_support(video['decoded_end_s'],media[8]['duration_s'],video['source_fps'],len(media[8]['images']))
        metrics=dict(metrics,input_coverage_validator=VALIDATOR_VERSION,cpu_decoder_revision=2,decoder_owner_gpu=owner)
        if metrics.get('media_cache_hit'):
            metrics.update(rgb_reuse_requested=bool(reuse),input_rgb_converter={'path':'exact_media_cache'})
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
