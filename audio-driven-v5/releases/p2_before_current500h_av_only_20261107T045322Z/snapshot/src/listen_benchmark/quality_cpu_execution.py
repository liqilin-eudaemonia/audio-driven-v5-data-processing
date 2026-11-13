"""Bound MediaPipe executor threads without altering model/options or sampling."""
from contextlib import contextmanager


def serial_landmarker(base):
    class SerialFaceLandmarker(base):
        def __init__(self, graph_config, running_mode, packet_callback):
            graph_config.num_threads=1
            super().__init__(graph_config,running_mode,packet_callback)
    return SerialFaceLandmarker


@contextmanager
def bounded_detector_threads(root):
    from listen_visual_quality.measurement import backend
    vendor,*_=backend(root)
    previous=vendor.FaceLandmarker
    vendor.FaceLandmarker=serial_landmarker(previous)
    try:yield
    finally:vendor.FaceLandmarker=previous
