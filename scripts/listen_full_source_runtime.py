"""Execution acceleration only: NVDEC decode, original software scale/scene rule.

GPU resizing is deliberately not used: the reviewed CPU scaler and scene
threshold remain unchanged. Unsupported GPU decode falls back to the original
CPU path; it never turns a decoding failure into a no-cut result.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import listen_local_recovery as policy
from p3_listen.common import read, write

REVISION='nvdec-original-scaler-scene-cache-1'
CPU_DETECT=policy.detect_shots
CACHE=None
DECODERS_PER_GPU=4


@contextlib.contextmanager
def decoder_slot():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').replace(',','_')
    folder=Path('/tmp/listen-local-full-decoder-slots');folder.mkdir(exist_ok=True)
    held=None
    while held is None:
        for slot in range(DECODERS_PER_GPU):
            f=(folder/f'gpu_{gpu}_{slot}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            held=f;break
        if held is None:time.sleep(.1)
    try:yield
    finally:held.close()


def detect_shots(source):
    src=source['source'];stat=Path(src['archive']).stat()
    identity=dict(source=src,archive_size=stat.st_size,archive_mtime_ns=stat.st_mtime_ns,
                  revision=REVISION,duration=source['duration'])
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=CACHE/(key+'.json') if CACHE is not None else None
    if path is not None and path.exists():
        cached=read(path)
        if cached['identity']==identity:return cached['cuts']
    start=time.monotonic();fallback=None
    with decoder_slot():
        result=subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-filter_threads','1',
            '-noautorotate','-hwaccel','cuda','-hwaccel_output_format','cuda','-hwaccel_device','0',
            '-i',policy.old.base.media.uri(src),'-vf',
            'hwdownload,format=nv12,format=yuv420p,scale=320:-2,select=gt(scene\\,0.35),metadata=print:file=-',
            '-an','-vsync','0','-f','null','-'],capture_output=True,text=True)
    if result.returncode:
        fallback=result.stderr[-1600:]
        cuts=CPU_DETECT(source)
    else:
        cuts=[float(t) for t in re.findall(r'pts_time:([0-9.]+)',result.stdout)]
        cuts=[t for t in cuts if .1<t<source['duration']-.1]
    if path is not None:
        write(path,dict(identity=identity,cuts=cuts,elapsed_seconds=time.monotonic()-start,
                        decoder='cpu_fallback' if fallback else 'nvdec',fallback_reason=fallback))
    return cuts


def install(cache):
    global CACHE
    CACHE=Path(cache);CACHE.mkdir(parents=True,exist_ok=True)
    policy.detect_shots=detect_shots
