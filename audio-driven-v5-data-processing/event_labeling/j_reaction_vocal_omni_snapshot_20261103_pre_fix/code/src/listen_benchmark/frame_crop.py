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
    _conversion_state.last_format = frame.format.name
    if frame.format.name not in ('yuv420p', 'yuv420p10le', 'rgb24'):
        _conversion_state.last_path = 'legacy_format_fallback'
        return crop_original_frame(frame, transform)
    _conversion_state.last_path = 'thread_local_full_rgb_v1'
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


def reused_conversion_info():
    return {'path': getattr(_conversion_state, 'last_path', None),
            'pixel_format': getattr(_conversion_state, 'last_format', None),
            'swscale_threads': 1 if getattr(_conversion_state, 'supports_threads', None) else 'legacy_default'}


def yuv_crop_geometry(transform):
    """32-aligned context preserves chroma phase and ordered-dither phase."""
    validate_transform(transform)
    w,h=transform['source_size'];c=transform['crop']
    x=max(0,(c['x']-32)//32*32);y=max(0,(c['y']-32)//32*32)
    right=min(w,((c['x']+c['width']+63)//32)*32)
    bottom=min(h,((c['y']+c['height']+63)//32)*32)
    local=dict(transform,source_size=[right-x,bottom-y],crop=dict(c,x=c['x']-x,y=c['y']-y))
    return (x,y,right,bottom),local


def crop_yuv_with_halo(frame, transform):
    """Copy native planes around the crop; never rescale YUV or change range."""
    import av
    validate_transform(transform)
    if [frame.width,frame.height]!=transform['source_size']:
        raise ValueError('Source dimensions differ')
    if frame.format.name not in ('yuv420p','yuv420p10le') or frame.width%2 or frame.height%2:
        return frame,transform
    (x,y,right,bottom),local=yuv_crop_geometry(transform)
    # libavfilter preserves *all* AVFrame properties, including chroma_location
    # which is not exposed by PyAV. Rebuilding a frame from planes loses it and
    # changes real decoder RGB pixels, even with colorspace/range copied.
    key=(frame.width,frame.height,frame.format.name,str(frame.time_base),x,y,right,bottom)
    if getattr(_conversion_state,'crop_graph_key',None)!=key:
        graph=av.filter.Graph()
        if hasattr(graph,'threads'):graph.threads=1
        src=graph.add_buffer(template=frame)
        crop=graph.add('crop',f'{right-x}:{bottom-y}:{x}:{y}:exact=1')
        sink=graph.add('buffersink');src.link_to(crop);crop.link_to(sink);graph.configure()
        _conversion_state.crop_graph_key=key
        _conversion_state.crop_graph=(graph,src,sink)
    graph,src,sink=_conversion_state.crop_graph
    src.push(frame)
    return sink.pull(),local


def crop_original_frame_roi(frame,transform):
    cropped,local=crop_yuv_with_halo(frame,transform)
    return crop_original_frame_reused(cropped,local)
