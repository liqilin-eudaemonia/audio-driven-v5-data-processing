"""Shared-file handoff at the next completed GPU batch; no process signals."""
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

WORK=Path(__file__).resolve().parents[1]/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111'


def save(name,value):
    f=WORK/name;tmp=f.with_name(f.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(f)


def main():
    marker=WORK/'STREAM_STOP';token=uuid.uuid4().hex
    with marker.open('x') as stream:json.dump(dict(owner_script=Path(__file__).name,token=token,reason='Complete current GPU batch then adopt resource profilev3'),stream)
    version=marker.stat().st_mtime_ns
    save('STREAM_FILE_HANDOFF.json',dict(at=time.time(),pid=os.getpid(),host=os.uname().nodename,token=token,
        state='waiting_for_current_batch_and_decoder',no_GPU_process_signal=True))
    end=time.time()+3600
    while time.time()<end:
        if not marker.exists() or marker.stat().st_mtime_ns!=version or json.loads(marker.read_text()).get('token')!=token:
            save('STREAM_FILE_HANDOFF_ABORTED.json',dict(at=time.time(),reason='Stop marker changed by another actor'));return
        with (WORK/'stream_controller.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                if marker.stat().st_mtime_ns==version and json.loads(marker.read_text()).get('token')==token:
                    marker.unlink()
                    save('STREAM_FILE_HANDOFF_COMPLETE.json',dict(at=time.time(),state='Original controller exited safely; on-node watchdog starts revised controller',
                        all_committed_outputs_preserved=True,no_GPU_process_signal=True))
                return
        time.sleep(15)
    if marker.exists() and marker.stat().st_mtime_ns==version and json.loads(marker.read_text()).get('token')==token:marker.unlink()
    save('STREAM_FILE_HANDOFF_TIMEOUT.json',dict(at=time.time(),reason='No completed controller boundary within1h; own stop removed, work continues'))


if __name__=='__main__':main()
