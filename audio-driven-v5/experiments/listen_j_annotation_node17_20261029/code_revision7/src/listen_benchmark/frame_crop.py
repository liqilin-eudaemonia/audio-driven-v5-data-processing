"""Exact RGB crop without constructing a full-resolution PIL image."""
from PIL import Image
from listen_labeling.media import validate_transform


def crop_original_frame(frame, transform):
    """Keep RGB conversion and PIL bicubic semantics; copy only the crop to PIL."""
    validate_transform(transform)
    if [frame.width,frame.height]!=transform['source_size']:
        raise ValueError('Decoded source size mismatch; cannot crop a pre-cropped video again')
    rgb=frame.to_ndarray(format='rgb24')
    c=transform['crop']
    cropped=rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]
    return Image.fromarray(cropped).resize((512,512),resample=Image.Resampling.BICUBIC)
