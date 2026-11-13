"""Honor the user's stop-after-VQ request without touching other GPU jobs."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/common_method_comparison_20261106'


def read(p):return json.loads(p.read_text())


def main():
    run=Path(read(WORK/'ACTIVE_L2L_RUN.json')['path'])
    assert read(run/'STOP_AFTER_VQ_REQUEST.json')['user_requested']
    uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
    pid=int(read(run/'LAUNCH.json')['pid'])
    command=Path('/proc')/str(pid)/'cmdline'
    def target_alive():
        try:return b'vico_l2l_retrain_v2.py' in command.read_bytes()
        except FileNotFoundError:return False
    deadline=time.monotonic()+3600
    while not (run/'vq_COMPLETE.json').exists():
        if not target_alive():raise RuntimeError('Training exited before VQ completion; inspect run.log')
        if time.monotonic()>deadline:raise TimeoutError('VQ has not completed within one hour; no unrelated jobs were touched')
        time.sleep(.05)
    completion=read(run/'vq_COMPLETE.json')
    if target_alive():
        assert os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGTERM)
    # Keep the complete-stage receipt, selected checkpoint and all caches.
    for _ in range(60):
        if not target_alive():break
        time.sleep(.1)
    if target_alive():
        os.killpg(pid,signal.SIGKILL)
    state=''
    for _ in range(40):
        state=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        used,util=map(lambda x:int(x.strip()),state.split(','))
        if used<100:break
        time.sleep(.25)
    app_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).splitlines()
    current_jobs=[dict(pid=int(row.split(',')[1].strip()),memory=row.split(',')[2].strip()) for row in app_rows if uuid in row]
    own_context_released=not target_alive() and all(row['pid']!=pid for row in current_jobs)
    result=dict(status='paused_after_vq',requested_by_user=True,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host='172.29.189.119',physical_gpu=7,
        controller_pid=pid,controller_exited=not target_alive(),vq=completion,
        predictor_started=(run/'predictor_history.json').exists(),test_started=(run/'TEST_GENERATION_COMPLETE.json').exists(),
        gpu_state=state,gpu_memory_mib=used,gpu_utilization_percent=util,gpu_released=own_context_released,
        gpu_currently_idle=used<100,current_gpu_jobs=current_jobs,
        preserved=['VQ selected checkpoint','VQ history','397-pair feature cache','training statistics','split and protocol'])
    tmp=run/'PAUSED_AFTER_VQ.tmp.json';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(run/'PAUSED_AFTER_VQ.json')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert result['controller_exited'] and result['gpu_released']
    assert not result['predictor_started'] and not result['test_started']


if __name__=='__main__':main()
