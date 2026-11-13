"""Bounded lossless YUV handoff; each consumer keeps its own RGB converter."""
from pathlib import Path
import hashlib
import numpy as np
from .j_annotation import read, write

SCHEMA = 'j-shared-native-yuv10-roi-v1'
MAX_BYTES = 4 * 1024 ** 3


class RawWriter:
    def __init__(self, directory, source):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.frames = []
        self.digest = hashlib.sha256()
        self.size = 0
        self.valid = True
        (self.directory / 'planes.bin').write_bytes(b'')

    def append(self, frame, t):
        if not self.valid:
            return
        if frame.format.name != 'yuv420p10le':
            self.valid = False
            return
        from .frame_crop import crop_yuv_with_halo
        frame,local=crop_yuv_with_halo(frame,self.source['video_transform'])
        record = dict(time=t, width=frame.width, height=frame.height,
                      colorspace=frame.colorspace, color_range=frame.color_range, planes=[],transform=local)
        with (self.directory / 'planes.bin').open('ab') as output:
            for plane in frame.planes:
                raw = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
                packed = np.ascontiguousarray(raw[:, :plane.width * 2])
                if self.size + packed.nbytes > MAX_BYTES:
                    self.valid = False
                    return
                record['planes'].append(dict(offset=self.size, width=plane.width,
                                             height=plane.height, bytes=packed.nbytes))
                output.write(packed)
                self.digest.update(packed)
                self.size += packed.nbytes
        self.frames.append(record)

    def finish(self, reader):
        if not self.valid or not self.frames:
            return False
        times = [f['time'] for f in self.frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Nonmonotonic raw shared PTS')
        write(self.directory / 'manifest.json', dict(schema=SCHEMA, source=self.source,
              frames=self.frames, reader=reader, bytes=self.size, sha256=self.digest.hexdigest()))
        return True


def load_raw(directory, source, manifest):
    """Convert reconstructed original planes using the nod runtime, not Omni's."""
    import av
    from .frame_crop import crop_original_frame,yuv_crop_geometry
    if manifest['schema'] != SCHEMA or manifest['source'] != source:
        raise ValueError('Raw shared source mismatch')
    path = Path(directory) / 'planes.bin'
    if path.stat().st_size != manifest['bytes'] or not 0 < manifest['bytes'] <= MAX_BYTES:
        raise ValueError('Raw shared byte count mismatch')
    data = np.memmap(path, mode='r', dtype=np.uint8)
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('Raw shared bytes changed')
    images, times, offset = [], [], 0
    for record in manifest['frames']:
        transform=source['video_transform']
        if not any(n%2 for n in transform['source_size']):
            _,transform=yuv_crop_geometry(transform)
        if record['transform']!=transform or [record['width'], record['height']] != transform['source_size']:
            raise ValueError('Raw shared dimensions changed')
        frame = av.VideoFrame(record['width'], record['height'], 'yuv420p10le')
        frame.colorspace = record['colorspace']
        frame.color_range = record['color_range']
        if len(record['planes']) != len(frame.planes):
            raise ValueError('Raw shared plane count mismatch')
        for plane, rec in zip(frame.planes, record['planes']):
            size = plane.width * plane.height * 2
            if (rec['width'], rec['height'], rec['bytes'], rec['offset']) != (plane.width, plane.height, size, offset):
                raise ValueError('Raw shared plane layout mismatch')
            target = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
            target[:] = 0
            target[:, :plane.width * 2] = data[offset:offset + size].reshape(plane.height, plane.width * 2)
            offset += size
        images.append(np.asarray(crop_original_frame(frame, transform)))
        times.append(record['time'])
    if offset != len(data) or not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Raw shared support mismatch')
    return np.stack(images), times, manifest['reader']
