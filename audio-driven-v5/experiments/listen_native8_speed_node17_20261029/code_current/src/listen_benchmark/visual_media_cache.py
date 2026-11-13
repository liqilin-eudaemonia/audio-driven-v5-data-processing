"""Optional node-local cache of exact decoded PIL frames (trusted local files)."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import time


def cached_media(source, directory, decode):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    # This cache is created by our own process; never accept downloaded pickles.
    stat=Path(source['source']['archive']).stat()
    identity={'source':source,'archive_size':stat.st_size,'archive_mtime_ns':stat.st_mtime_ns,
              'decoder_sha256':hashlib.sha256(Path(__file__).with_name('visual_native8.py').read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    target=directory/f'{key}.pickle'
    with (directory/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if target.exists():
            tick=time.monotonic()
            with target.open('rb') as f:whole=pickle.load(f)
            return whole,{'decode_s':0.,'decode_lock_wait_s':0.,'media_cache_hit':True,
                          'cache_read_s':time.monotonic()-tick}
        whole,metrics=decode()
        temp=target.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with temp.open('wb') as f:pickle.dump(whole,f,protocol=pickle.HIGHEST_PROTOCOL)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        return whole,dict(metrics,media_cache_hit=False,cache_read_s=0.)
