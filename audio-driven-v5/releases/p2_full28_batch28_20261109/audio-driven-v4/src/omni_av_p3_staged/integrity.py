"""Immutable input fingerprints, separate from the expensive planner readouts."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import time

import torch

from omni_av_p2.checkpoint import atomic_json_new
from .orchestration import read, sha, stopped, write

VERSION = 'p3.input-integrity.v1'


def key(entry):
    return entry['split'] + ':' + entry['sample_id']


def record_path(root, entry):
    sid=entry['sample_id']
    return Path(root)/entry['split']/sid[:3]/(sid+'.json')


def source_paths(entry, data_root):
    base=Path(data_root)/'ltx23_precomputed_720p_area_qualified_20260920'
    paths={'text':str(Path(entry['text']).resolve())}
    for name in ('latents','audio_latents'):
        choices=[base/name/(entry['sample_id']+'.pt'),base/name/entry['split']/(entry['sample_id']+'.pt')]
        found=[p for p in choices if p.is_file()]
        if len(found)!=1:raise ValueError('VAE source path mismatch')
        paths[name]=str(found[0].resolve())
    return paths


def fingerprint(entry, data_root, manifest_sha):
    files={}
    for name,path in source_paths(entry,data_root).items():
        before=Path(path).stat()
        digest=sha(path)
        after=Path(path).stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Input changed during fingerprinting')
        files[name]=dict(path=path,sha256=digest)
    return dict(version=VERSION,manifest_sha256=manifest_sha,sample=key(entry),files=files)


def pin_entry(root, entry, data_root, manifest_sha):
    """Engineering tests may pin their subset before the full seal exists."""
    path=record_path(root,entry)
    if path.exists():
        value=read(path)
    else:
        value=fingerprint(entry,data_root,manifest_sha)
        path.parent.mkdir(parents=True,exist_ok=True)
        try:atomic_json_new(path,value)
        except FileExistsError:
            if read(path)!=value:raise ValueError('Concurrent input fingerprint mismatch')
    if (value.get('version')!=VERSION or value.get('manifest_sha256')!=manifest_sha
            or value.get('sample')!=key(entry)
            or {k:v['path'] for k,v in value['files'].items()}!=source_paths(entry,data_root)):
        raise ValueError('Input fingerprint identity mismatch')
    return value


@lru_cache(maxsize=4)
def _seal(path, mtime_ns, size):
    return read(path)


def load_record(root, entry, manifest_sha, data_root, allow_unsealed=False):
    root=Path(root);seal_path=root/'complete.json'
    if not seal_path.exists():
        if allow_unsealed:return pin_entry(root,entry,data_root,manifest_sha)
        raise ValueError('Full input-integrity seal required')
    stat=seal_path.stat();seal=_seal(str(seal_path),stat.st_mtime_ns,stat.st_size)
    path=record_path(root,entry)
    if (seal.get('version')!=VERSION or seal.get('manifest_sha256')!=manifest_sha
            or seal['entries'].get(key(entry))!=sha(path)):
        raise ValueError('Input fingerprint seal mismatch')
    return pin_entry(root,entry,data_root,manifest_sha)


def verified_load(path, files, map_location='cpu'):
    """Hash exactly the bytes deserialized; no check/load race or duplicate read."""
    path=str(Path(path).resolve())
    expected=[v['sha256'] for v in files.values() if v['path']==path]
    if len(expected)!=1:raise ValueError('Unbound input path')
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected[0]:raise ValueError('Input content hash mismatch: '+path)
    return torch.load(io.BytesIO(payload),map_location=map_location,weights_only=True)


def prepare_inputs(manifest, root, manifest_sha, control_root, workers=4):
    root=Path(root);entries=manifest['train']+manifest['dev'];root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():
        seal=read(root/'complete.json')
        if seal['manifest_sha256']!=manifest_sha or seal['version']!=VERSION:raise ValueError('Input seal mismatch')
        return seal
    hashes={};begin=time.time()
    def one(entry):
        if stopped(control_root):raise InterruptedError('User STOP')
        existed=record_path(root,entry).exists()
        record=pin_entry(root,entry,manifest['config']['data_root'],manifest_sha)
        # On a resumed preparation, refuse to repin changed source data.
        if existed:
            for file in record['files'].values():
                if sha(file['path'])!=file['sha256']:raise ValueError('Input content hash mismatch: '+file['path'])
        return key(entry),sha(record_path(root,entry))
    # Bound submissions: Python 3.13 Executor.map eagerly queues all 91k items.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator=iter(entries);pending=[]
        for _ in range(workers):
            entry=next(iterator,None)
            if entry is not None:pending.append(pool.submit(one,entry))
        while pending:
            name,digest=pending.pop(0).result();hashes[name]=digest
            if len(hashes)%100==0 or len(hashes)==len(entries):
                write(root/'progress.json',dict(completed=len(hashes),total=len(entries),time=time.time(),started=begin))
            entry=next(iterator,None)
            if entry is not None:pending.append(pool.submit(one,entry))
    seal=dict(version=VERSION,manifest_sha256=manifest_sha,entries=hashes,
              train=len(manifest['train']),dev=len(manifest['dev']),count=len(entries))
    atomic_json_new(root/'complete.json',seal)
    return seal
