"""Optional CPU preparation: share exact frame arrays across overlapping windows.

Callers must retain the returned entries until inference completes. No production
worker enables this path by default. Source images and metadata are not mutated.
"""
import math

from .full_raw import views
from .visual_native8 import prepare_input


def prepare_shared_inputs(processor, whole):
    import numpy as np
    media = whole[8]
    frames = np.stack([np.asarray(image) for image in media['images']])
    prepared = {}
    for view, start, end in views(media['duration_s']):
        if view == 'full':
            selected = frames
        else:
            first = round(start * 8)
            last = min(len(frames), math.ceil(end * 8 - 1e-8))
            selected = frames[first:last]
        prepared[view] = {'interval': (start, end),
                          'input': prepare_input(processor, whole, start, end, view,
                                                 frame_array=selected)}
    return prepared
