"""Preflight actual output and launch the separation-only final audit once."""
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

import finish_separation as final

WORK = Path(__file__).resolve().parent


def main():
    marker=WORK/'FINISH_LAUNCH.json'
    assert not marker.exists(), 'Already launched; inspect recorded PID'
    source=final.read(WORK/'RESOURCE_SCHEDULER_V2_LAUNCH.json')
    assert final.state(source['pid']) not in [None,'Z'], 'Verify separation manager before launching observer'
    actual=next((WORK/'execution').glob('worker_*/separated/*/*/complete.json'))
    receipt=final.read(actual)
    shard=actual.parent.parents[2]
    assert final.digest(shard/'PROTOCOL.json')==receipt['protocol_sha256']
    runtime=final.read(shard/'runtime_history'/(receipt['runtime_sha256']+'.json'))
    assert runtime['device']['scheduler_protocol_sha256']==final.digest(WORK/'RESOURCE_SCHEDULER_V2_PROTOCOL.json')
    raw=np.load(actual.parent/'estimate.npy',allow_pickle=False)
    wav,sr=sf.read(actual.parent/'target.wav',dtype='float32')
    sample=next(s for s in final.read(WORK/'PROTOCOL.json')['samples'] if s['id']==receipt['id'])
    mix,rate=sf.read(sample['mix'],dtype='float32')
    assert sr==rate==16000 and raw.shape==wav.shape==mix.shape==(sample['samples'],)
    discrepancies=[]
    for name,values in [('raw_audio_metrics',raw),('saved_audio_metrics',wav)]:
        for key,value in final.metrics(mix,values).items():
            expected=receipt[name][key]
            assert (value is None and expected is None) or (value is not None and expected is not None and abs(value-expected)<1e-7),(name,key,value,expected)
            if value is not None:discrepancies.append(abs(value-expected))
    with np.load(actual.parent/'mouthroi/speaker1.npz',allow_pickle=False) as data:
        assert data['data'].shape==(round(sample['samples']/16000*25),96,96)
    with np.load(actual.parent/'landmark/speaker1.npz',allow_pickle=True) as data:
        assert len(data['data'])==round(sample['samples']/16000*25)
    error=float(np.abs(raw-wav).max());assert error<=1e-6
    final.save(WORK/'FINISH_PREFLIGHT.json',dict(passed=True,actual_receipt=str(actual),
        receipt_sha256=final.digest(actual),audit_implementation_sha256=final.digest(WORK/'finish_separation.py'),
        float_WAV_max_abs_error=error,independent_metric_max_abs_error=max(discrepancies)))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    with (WORK/'finish_separation.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(WORK/'finish_separation.py'),'watch','--pid',str(source['pid'])],
            cwd=final.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    final.save(marker,dict(pid=child.pid,watched_pid=source['pid'],launched_unix=time.time(),
        audit_implementation_sha256=final.digest(WORK/'finish_separation.py'),CPU_only=True,
        continuation='audit separation artifacts and exit; no AF commands'))
    print(child.pid,flush=True)


if __name__=='__main__':main()
