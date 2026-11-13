"""Persistent, bounded CPU-only decoder; never inherit the training process state."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image


class IsolatedDecoder:
    def __init__(self, threads=4, timeout=30, log_dir=None, command=None):
        self.threads = threads
        self.timeout = timeout
        self.pid = os.getpid()
        self.temp = tempfile.TemporaryDirectory(prefix='planning-first-frame-')
        self.process = None
        self.command = command or [sys.executable, '-m', 'p2_planning_sft.isolated_decode']
        self.log_dir = Path(log_dir) if log_dir else Path(self.temp.name)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stderr = open(self.log_dir / f'decoder-{os.uname().nodename}-{self.pid}.log', 'a', buffering=1)
        atexit.register(self.close)

    def stop(self):
        p, self.process = self.process, None
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
            p.stdin.close()
            p.stdout.close()

    def close(self):
        if os.getpid() == self.pid:
            self.stop()
            self.temp.cleanup()
            self.stderr.close()

    def start(self):
        env = dict(os.environ, PYAV_LOGGING='off', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        src = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, env=env, bufsize=0)

    def __call__(self, source):
        if os.getpid() != self.pid:
            raise RuntimeError('Decoder cannot be inherited across workers')
        output = Path(self.temp.name) / 'frame.npy'
        errors = []
        for mode in ('auto', 'serial'):
            try:
                if self.process is None or self.process.poll() is not None:
                    self.stop()
                    self.start()
                request = dict(source=source, threads=self.threads, mode=mode, output=str(output))
                payload = (json.dumps(request) + '\n').encode()
                # Requests contain only source metadata, never media bytes.
                self.process.stdin.write(payload)
                ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
                if not ready:
                    raise TimeoutError(f'{mode} decoder exceeded {self.timeout}s')
                reply = json.loads(self.process.stdout.readline())
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                frame = np.load(output, allow_pickle=False)
                if frame.shape != (512, 512, 3) or frame.dtype != np.uint8:
                    raise ValueError('Invalid decoded RGB dimensions/type')
                if hashlib.sha256(frame.tobytes()).hexdigest() != reply['rgb_sha256']:
                    raise ValueError('Decoded RGB transfer checksum mismatch')
                output.unlink()
                return Image.fromarray(frame), reply['pts']
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                errors.append(str(exc))
                self.stderr.write(json.dumps(dict(time=time.time(), mode=mode, error=str(exc), source=source)) + '\n')
                self.stderr.flush()
                self.stop()
                output.unlink(missing_ok=True)
        raise RuntimeError('First-frame decoding failed without skipping sample: ' + '; '.join(errors))


def serve():
    # Linux parent-death signal also terminates a child stuck in native decoding.
    import ctypes
    import faulthandler
    import signal
    import traceback
    parent = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError('Cannot enable decoder parent-death cleanup')
    if os.getppid() != parent or parent == 1:
        return
    from p2_planning_sft.fast_media import decode_first_frame
    from listen_response_sft.data import first_frame
    for line in sys.stdin:
        request = json.loads(line)
        faulthandler.dump_traceback_later(20, file=sys.stderr)
        try:
            if request['mode'] == 'auto':
                image, pts = decode_first_frame(request['source'], request['threads'])
            else:
                image, pts = first_frame(request['source'])
            array = np.asarray(image)
            output = Path(request['output'])
            partial = output.with_suffix('.partial')
            with partial.open('wb') as f:
                np.save(f, array, allow_pickle=False)
            os.replace(partial, output)
            reply = dict(pts=pts, rgb_sha256=hashlib.sha256(array.tobytes()).hexdigest())
        except Exception:
            reply = dict(error=traceback.format_exc())
        finally:
            faulthandler.cancel_dump_traceback_later()
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    serve()
