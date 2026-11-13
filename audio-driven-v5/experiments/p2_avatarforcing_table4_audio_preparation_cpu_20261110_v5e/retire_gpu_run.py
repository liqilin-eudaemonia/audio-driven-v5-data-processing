"""After CPU parity passes, retire only this session's old separation jobs."""
import json
import os
from pathlib import Path
import signal
import time

import parallel

WORK=Path(__file__).resolve().parent
OLD=WORK.parent/'p2_avatarforcing_table4_audio_preparation_cpu_frontend_20261110_v5d'


def process(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    return dict(pid=pid,state=(p/'stat').read_text().rsplit(')',1)[1].split()[0],
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
        children=(p/'task'/str(pid)/'children').read_text().split())


def main():
    target=WORK/'OLD_GPU_RETIREMENT.json'
    assert not target.exists(), 'Already retired; inspect the receipt'
    assert parallel.read(WORK/'PARALLEL_PARITY.json')['passed']
    cpu=parallel.read(WORK/'RUN_STATUS.json')
    assert cpu['stage']=='running_full86_CPU' and process(cpu['pid'])['state']!='Z'
    assert process(cpu['manager_pid'])['state']!='Z'
    manager_pid=parallel.read(OLD/'RESOURCE_SCHEDULER_V2_LAUNCH.json')['pid']
    observer_pid=parallel.read(OLD/'FINISH_LAUNCH.json')['pid']
    stopped=[]
    try:
        manager=process(manager_pid)
        assert manager and str(OLD/'resource_manager_v2.py') in manager['argv']
        os.kill(manager_pid,signal.SIGSTOP);stopped.append(manager_pid)
        time.sleep(.1)
        manager=process(manager_pid)
        workers=[]
        for value in manager['children']:
            child=process(int(value))
            if child is None or child['state']=='Z':continue
            if str(OLD/'resource_execute.py') in child['argv']:
                os.kill(child['pid'],signal.SIGSTOP);stopped.append(child['pid']);workers.append(child)
            else:
                assert 'nvidia-smi' in child['argv'], child
        observer=process(observer_pid)
        if observer and observer['state']!='Z':
            assert str(OLD/'finish_separation.py') in observer['argv']
            os.kill(observer_pid,signal.SIGSTOP);stopped.append(observer_pid)
        receipts=list((OLD/'execution').glob('worker_*/separated/*/*/complete.json'))
        validated=[]
        for path in receipts:
            r=parallel.read(path)
            for filename,digest in r['files'].items():assert parallel.sha(filename)==digest
            validated.append(dict(id=r['id'],role=r['role'],receipt=str(path),receipt_sha256=parallel.sha(path)))
        before=parallel.read(OLD/'PARALLEL_STATUS.json')
        for pid in reversed(stopped):
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(100):
            if all(process(pid) is None or process(pid)['state']=='Z' for pid in stopped):break
            time.sleep(.1)
        else:raise RuntimeError('Some old owned processes have not exited')
        result=dict(passed=True,reason='CPU four-process parity passed; full86 uniform CPU run active; retire only our redundant GPU job',
            old_controller=manager,old_workers=workers,old_observer=observer,terminated_pids=stopped,
            validated_legacy_streams=validated,legacy_before_status=before,new_CPU_runner_pid=cpu['pid'],
            new_CPU_manager_pid=cpu['manager_pid'],all_terminated_processes_verified_owned=True,
            other_jobs_touched=False,retired_unix=time.time())
        parallel.save(target,result)
        parallel.save(OLD/'PARALLEL_STATUS.json',dict(stage='superseded_by_uniform_CPU_batch',completed=len(validated),total=86,
            legacy_partial_outputs_retained=True,retirement_receipt=str(target),new_work=str(WORK),updated_unix=time.time()))
        parallel.save(OLD/'FINISH_STATUS.json',dict(stage='superseded_by_uniform_CPU_batch',retirement_receipt=str(target),updated_unix=time.time()))
        print('Retired owned GPU processes:',stopped,'valid partial streams preserved:',len(validated),flush=True)
    except BaseException:
        for pid in stopped:
            p=process(pid)
            if p and p['state']=='T':os.kill(pid,signal.SIGCONT)
        raise


if __name__=='__main__':main()
