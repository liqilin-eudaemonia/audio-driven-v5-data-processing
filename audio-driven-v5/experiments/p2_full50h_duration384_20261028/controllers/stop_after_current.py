"""Operational guard: stop scheduling once the requested final is durable."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

HERE = Path(__file__).resolve().parent


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    request = json.loads((HERE / 'STOP_AFTER_CURRENT.json').read_text())
    checkpoint = Path(request['checkpoint'])
    marker = checkpoint.with_suffix('.complete.json')
    host = socket.gethostname()
    receipt = HERE / f'stop_after_current_{host}.json'
    heartbeat = 0
    while True:
        if marker.exists() and checkpoint.exists():
            complete = json.loads(marker.read_text())
            if complete['step'] != request['budget']:
                raise RuntimeError('Unexpected final checkpoint step')
            # Publish the scheduling stop before expensive checksum verification.
            # A complete marker is written only after the checkpoint and its hash.
            for root in request['stop_roots']:
                Path(root, 'STOP').write_text(
                    'User requested stop after completed AV+Event+Feature S1.\n')
            digest = hashlib.sha256()
            with checkpoint.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            valid = digest.hexdigest() == complete['sha256']
            atomic(receipt, dict(at=time.time(), pid=os.getpid(), host=host,
                                status='stopped_after_final' if valid else 'checksum_failed_stopped',
                                checkpoint=str(checkpoint), step=complete['step'],
                                sha256=digest.hexdigest(), checksum_valid=valid))
            return
        if time.monotonic() - heartbeat > 10:
            atomic(receipt, dict(at=time.time(), pid=os.getpid(), host=host,
                                status='armed_waiting_final', checkpoint=str(checkpoint)))
            heartbeat = time.monotonic()
        time.sleep(0.1)


if __name__ == '__main__':
    main()
