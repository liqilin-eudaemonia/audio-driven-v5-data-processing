"""Shared per-sample inference/export claims. No sampling or media math changes."""
import fcntl
import hashlib
import json
from pathlib import Path


def all_items(work, fallback, protocol):
    if not protocol.get('shared_item_queue'):return fallback
    path=Path(protocol['shared_items_manifest'])
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=protocol['shared_items_sha256']:
        raise ValueError('Fixed shared inference item list changed')
    items=json.loads(payload)
    if len(items)!=protocol['total_samples'] or len({x['key'] for x in items})!=len(items):
        raise ValueError('Shared queue duplicate or missing items')
    return sorted(items,key=lambda x:(-x['geometry']['native_frames'],x['key']))


def claim_items(work, fallback, protocol, arm, stage):
    """Hold an exclusive sample claim across preparation, inference and commit.

    Native output commits remain the source of truth. Completed media are never
    regenerated; a process death releases its claim so another GPU can resume.
    """
    if not protocol.get('shared_item_queue'):
        yield from fallback
        return
    work=Path(work)
    from .runtime import ready
    for item in all_items(work,fallback,protocol):
        if (work/'STOP').exists():return
        generated=work/'generated'/arm['name']/item['key']
        target=work/('latents' if stage=='render' else 'generated')/arm['name']/item['key']
        if ready(generated) or ready(target):continue
        if stage=='decode' and not ready(work/'latents'/arm['name']/item['key']):continue
        key=hashlib.sha256(item['key'].encode()).hexdigest()
        path=work/'sample_claims'/stage/arm['name']/key[:2]/(key+'.lock')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            try:
                if ready(generated) or ready(target):continue
                yield item
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
