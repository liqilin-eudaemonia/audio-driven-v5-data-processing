"""Opt-in NVDEC for 8-bit H264, preserving the native8 CPU input protocol.

No production worker enables this module by default. Unsupported formats and
hardware failures use the original CPU path. No transcoding or altered sampling.
"""
from contextlib import contextmanager

from .full_raw import SourceBytes, open_source
from .visual_media_coverage import prepare_media_covered


@contextmanager
def open_cuda_source(source, device):
    import av
    from av.codec.hwaccel import HWAccel
    from listen_labeling.full_source import source_uri
    settings = HWAccel('cuda', device=device, allow_software_fallback=False)
    try:
        container = av.open(source_uri(source), timeout=(30, 60), hwaccel=settings)
    except av.error.InvalidDataError:
        with SourceBytes(source) as member, av.open(member, hwaccel=settings) as container:
            yield container, 'bounded_original_bytes'
    else:
        with container:
            yield container, 'ffmpeg_subfile'


def prepare_hybrid_media(row, device='0', fast_crop=True):
    """Return exact covered media plus explicit hardware/fallback provenance.

    Device is an index within the decoder process's CUDA_VISIBLE_DEVICES mask.
    Source probing creates no CUDA context for unsupported codecs. A failed
    hardware attempt is discarded completely before CPU recovery.
    """
    from .frame_crop import crop_original_frame
    from listen_labeling.media import crop_resize
    if not isinstance(device, str) or not device.isdecimal():
        raise ValueError('An explicit local CUDA device index is required')
    info = {'cuda_attempted': False, 'cuda_used': False, 'cpu_fallback': False,
            'cuda_device': device, 'pixel_normalization': None}

    @contextmanager
    def opener(source):
        with open_source(source) as (container, reader):
            stream = container.streams.video[0]
            fmt = stream.codec_context.format
            info.update(source_codec=stream.codec_context.name,
                        source_profile=stream.codec_context.profile,
                        source_pixel_format=fmt.name if fmt is not None else None)
            supported = info['source_codec'] == 'h264' and info['source_pixel_format'] == 'yuv420p'
            if not supported:
                info['cpu_reason'] = 'format_outside_validated_cuda_path'
                yield container, reader
                return
        info['cuda_attempted'] = True
        with open_cuda_source(source, device) as pair:
            info['cuda_used'] = True
            yield pair

    def crop(frame, transform):
        if info['cuda_used']:
            if frame.format.name != 'nv12':
                raise ValueError('Hardware output is not the validated NV12 format')
            # Lossless chroma-layout conversion restores the CPU swscale path.
            # Direct NV12->RGB has different rounding and is not interchangeable.
            frame = frame.reformat(format='yuv420p')
            info['pixel_normalization'] = 'nv12_to_yuv420p_before_original_rgb'
        return (crop_original_frame(frame, transform) if fast_crop
                else crop_resize(frame.to_image(), transform))

    try:
        result = prepare_media_covered(row, frame_crop=crop, source_opener=opener)
    except Exception as exc:
        if not info['cuda_attempted'] or isinstance(exc, MemoryError):
            raise
        info.update(cuda_used=False, cpu_fallback=True, cuda_error=repr(exc),
                    cpu_reason='hardware_attempt_failed', pixel_normalization=None)
        result = prepare_media_covered(row, frame_crop=crop_original_frame if fast_crop else None)
    return result, info
