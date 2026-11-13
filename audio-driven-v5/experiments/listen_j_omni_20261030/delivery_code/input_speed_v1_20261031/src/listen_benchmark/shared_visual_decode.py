"""Ephemeral, source-bound RGB frames shared by Omni preparation and CPU nodding."""
from pathlib import Path
import hashlib
import numpy as np
from .j_annotation import read,write


def save_native(directory,source,images,times,reader):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not images or len(images)!=len(times):raise ValueError('Incomplete shared decode')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Nonmonotonic shared PTS')
    frames=np.stack(images)
    if frames.dtype!=np.uint8 or frames.shape[1:]!=(512,512,3):raise ValueError('Invalid shared RGB frames')
    np.save(directory/'rgb.npy',frames,allow_pickle=False)
    write(directory/'manifest.json',dict(schema='j-shared-native-rgb-v1',source=source,
        times=times,reader=reader,shape=list(frames.shape),sha256=hashlib.sha256(frames).hexdigest()))


def load_native(directory,source):
    directory=Path(directory);m=read(directory/'manifest.json')
    if m['schema']!='j-shared-native-rgb-v1' or m['source']!=source:raise ValueError('Shared decode source mismatch')
    frames=np.load(directory/'rgb.npy',mmap_mode='r',allow_pickle=False)
    if frames.dtype!=np.uint8 or list(frames.shape)!=m['shape'] or frames.shape[1:]!=(512,512,3):raise ValueError('Shared decode shape mismatch')
    if hashlib.sha256(frames).hexdigest()!=m['sha256']:raise ValueError('Shared decode bytes changed')
    if len(m['times'])!=len(frames) or any(b<=a for a,b in zip(m['times'],m['times'][1:])):raise ValueError('Shared decode PTS mismatch')
    return frames,m['times'],m['reader']
