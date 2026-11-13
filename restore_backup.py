#!/usr/bin/env python3
"""Verify this snapshot, or restore its backed-up files into a new directory."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib

ROOT = Path(__file__).resolve().parent

def chunks(record):
    decoder = zlib.decompressobj(31) if record.get('encoding') == 'gzip' else None
    for storage in record.get('storage', [record['path']]):
        path = ROOT / storage
        if ROOT not in path.resolve().parents:
            raise ValueError('Unsafe storage path: ' + storage)
        with path.open('rb') as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                if decoder:
                    block = decoder.decompress(block)
                if block:
                    yield block
    if decoder:
        tail = decoder.flush()
        if tail:
            yield tail
        if not decoder.eof:
            raise ValueError('Truncated gzip stream: ' + record['path'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--destination', type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve() if args.destination else None
    if dest:
        if dest.exists() and any(dest.iterdir()):
            parser.error('Destination must be empty; existing files are never overwritten.')
        dest.mkdir(parents=True, exist_ok=True)
    count = total = 0
    with gzip.open(ROOT / 'backup_metadata/included.jsonl.gz', 'rt', encoding='utf-8') as manifest:
        for line in manifest:
            r = json.loads(line)
            digest = hashlib.sha256()
            size = 0
            output = None
            target = None
            if dest:
                target = dest / r['path']
                if dest not in target.resolve().parents:
                    raise ValueError('Unsafe restore path: ' + r['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('xb')
            try:
                for block in chunks(r):
                    digest.update(block)
                    size += len(block)
                    if output:
                        output.write(block)
            finally:
                if output:
                    output.close()
            if size != r['size'] or digest.hexdigest() != r['sha256']:
                raise ValueError('Checksum mismatch: ' + r['path'])
            if target:
                target.chmod(r['mode'] & 0o777)
                os.utime(target, ns=(r['mtime_ns'], r['mtime_ns']))
            count += 1
            total += size
            if count % 1000 == 0:
                print('Verified', count, 'files;', total, 'bytes', flush=True)
    print('SUCCESS:', count, 'files;', total, 'bytes verified.', flush=True)

if __name__ == '__main__':
    main()
