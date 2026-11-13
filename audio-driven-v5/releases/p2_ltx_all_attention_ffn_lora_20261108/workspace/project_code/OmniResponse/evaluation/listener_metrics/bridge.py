"""Serial workers: persist models without mixing TensorFlow/FER environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import selectors
import time

ROOT=Path(__file__).resolve().parents[3]/'audio-driven-v5'


class Worker:
    def __init__(self,kind,folder,device="cpu"):
        runtime=json.loads((ROOT/'configs/p2_omni_runtime.lock.json').read_text())
        python=ROOT/'.venv-p2-glare-detector/bin/python' if kind=='glare' else runtime['paths']['metrics_python']
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','') if device.startswith('cuda') else '',
            PYTHONPATH=str(Path(__file__).resolve().parents[2])+os.pathsep+str(ROOT/'src'))
        self.log=open(Path(folder)/(kind+'_worker.log'),'a')
        self.p=subprocess.Popen([str(python),'-m','evaluation.listener_metrics.worker','--kind',kind,'--device',device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,bufsize=0,start_new_session=True)
        self.buffer=b''
    def call(self,request,timeout=1800):
        self.p.stdin.write((json.dumps(request)+'\n').encode());self.p.stdin.flush()
        start=time.monotonic();selector=selectors.DefaultSelector();selector.register(self.p.stdout,selectors.EVENT_READ)
        try:
            while time.monotonic()-start<timeout:
                while b'\n' in self.buffer:
                    line,self.buffer=self.buffer.split(b'\n',1);line=line.decode(errors='replace');self.log.write(line+'\n');self.log.flush()
                    if line.startswith('LISTENER_RESULT '):
                        result=json.loads(line[len('LISTENER_RESULT '):])
                        if not result['ok']:raise RuntimeError(result['error'])
                        return result['result']
                if selector.select(timeout=1):
                    block=os.read(self.p.stdout.fileno(),65536)
                    if not block:raise RuntimeError('Metric worker exited; inspect log')
                    self.buffer+=block
            self.close();raise TimeoutError('Metric worker exceeded time limit')
        finally:selector.close()
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid,signal.SIGTERM)
                try:self.p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
        self.log.close()
