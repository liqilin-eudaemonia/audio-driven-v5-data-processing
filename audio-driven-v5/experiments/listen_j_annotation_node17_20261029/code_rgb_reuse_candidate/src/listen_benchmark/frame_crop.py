"""Exact RGB crop without constructing a full-resolution PIL image."""
from PIL import Image
import threading
from listen_labeling.media import validate_transform

_conversion_state = threading.local()


def crop_original_frame(frame, transform):
    """Keep RGB conversion and PIL bicubic semantics; copy only the crop to PIL."""
    validate_transform(transform)
    if [frame.width,frame.height]!=transform['source_size']:
        raise ValueError('Decoded source size mismatch; cannot crop a pre-cropped video again')
    rgb=frame.to_ndarray(format='rgb24')
    c=transform['crop']
    cropped=rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]
    return Image.fromarray(cropped).resize((512,512),resample=Image.Resampling.BICUBIC)


def crop_original_frame_reused(frame, transform):
    """Opt-in exact full-frame RGB conversion with one converter per thread.

    PyAV17 supports an explicit swscale thread count. PyAV16 uses its original
    defaults; it still benefits from retaining the converter across frames.
    Cropping stays after RGB conversion, including for 10-bit source frames.
    """
    from av.video.reformatter import VideoReformatter
    validate_transform(transform)
    if [frame.width, frame.height] != transform['source_size']:
        raise ValueError('Decoded source size mismatch; cannot crop a pre-cropped video again')
    if not hasattr(_conversion_state, 'converter'):
        _conversion_state.converter = VideoReformatter()
        _conversion_state.supports_threads = None
    converter = _conversion_state.converter
    if _conversion_state.supports_threads is False:
        converted = converter.reformat(frame, format='rgb24')
    else:
        try:
            converted = converter.reformat(frame, format='rgb24', threads=1)
            _conversion_state.supports_threads = True
        except TypeError as exc:
            if 'unexpected keyword' not in str(exc) or "'threads'" not in str(exc):
                raise
            _conversion_state.supports_threads = False
            converted = converter.reformat(frame, format='rgb24')
    rgb = converted.to_ndarray()
    c = transform['crop']
    cropped = rgb[c['y']:c['y'] + c['height'], c['x']:c['x'] + c['width']]
    return Image.fromarray(cropped).resize((512, 512), resample=Image.Resampling.BICUBIC)
