"""Exact-frame sequential decoder reuse inside one source job, with bounded cache."""
from contextlib import contextmanager
from types import SimpleNamespace


class Codec:
    def __init__(self, inner):self.inner=inner
    def __getattr__(self,name):return getattr(self.inner,name)
    @property
    def thread_count(self):return self.inner.thread_count
    @thread_count.setter
    def thread_count(self,value):
        # The frozen measurement entry sets this on every clip. PyAV forbids even an
        # identical assignment after opening; preserve the already configured value.
        if self.inner.is_open and self.inner.thread_count==value:return
        self.inner.thread_count=value


class Stream:
    def __init__(self, inner):self.inner=inner;self.codec_context=Codec(inner.codec_context)
    def __getattr__(self,name):return getattr(self.inner,name)


class Container:
    def __init__(self, av, reader, stats):
        self.reader=reader;self.inner=av.open(reader);self.stats=stats
        self.streams=SimpleNamespace(video=[Stream(s) for s in self.inner.streams.video])
        self.iterator=None;self.last=None;self.replay=False
        stats['opens']+=1

    def __enter__(self):return self
    def __exit__(self,*args):return False

    def seek(self, offset, *, stream, backward, any_frame):
        target=float(offset*stream.time_base)
        previous=float(self.last.pts*self.last.time_base) if self.last is not None else None
        # Reuse only forward, nearby requests. Older/overlapping or distant intervals seek normally.
        if (self.iterator is not None and previous is not None and
                0 <= target-previous <= 1.0):
            self.replay=True;self.stats['seeks_avoided']+=1
        else:
            self.inner.seek(offset,stream=stream.inner,backward=backward,any_frame=any_frame)
            self.iterator=iter(self.inner.decode(stream.inner));self.last=None;self.replay=False
            self.stats['seeks']+=1

    def decode(self, stream):
        if self.replay:
            self.replay=False
            yield self.last
        for frame in self.iterator:
            self.last=frame
            yield frame

    def close(self):
        self.inner.close();self.reader.close()


class AVSession:
    def __init__(self, av, source_type):
        self.av=av;self.source_type=source_type;self.cache={}
        self.stats={'opens':0,'seeks':0,'seeks_avoided':0}

    def open(self, source):
        key=(source.file.name,source.offset,source.size)
        if key not in self.cache:
            # Keep a separate reader alive: measure_row owns and closes its original SourceBytes.
            if len(self.cache)>=2:
                self.cache.pop(next(iter(self.cache))).close()
            reader=self.source_type({'archive':key[0],'offset':key[1],'byte_size':key[2]})
            self.cache[key]=Container(self.av,reader,self.stats)
        return self.cache[key]

    def close(self):
        for c in self.cache.values():c.close()


@contextmanager
def sequential_measurements(root):
    """Process-local adapter; detector lifecycle and pixel conversion remain unchanged."""
    from listen_visual_quality import measurement
    original=measurement.backend
    vendor,cv2,av,mp=original(root)
    session=AVSession(av,measurement.SourceBytes)
    measurement.backend=lambda _: (vendor,cv2,session,mp)
    try:yield session.stats
    finally:
        measurement.backend=original
        session.close()
