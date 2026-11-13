"""Authorized checkpoint6000 safe-stop gate before the four-node evaluation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save
WORK=ROOT/'experiments/p2_current_feature_whisper6000_eval32_20261108'
TRAIN=ROOT/'experiments/p2_current500h_av_whisper31_20261107'


def main():
    WORK.mkdir(exist_ok=True)
    with (WORK/'gate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'GATE_CONTROLLER.json',dict(pid=os.getpid(),at=time.time()))
        while True:
            if (WORK/'STOP').exists():return
            step=read(TRAIN/'status.json')['step']
            save(WORK/'GATE_STATUS.json',dict(stage='waiting_training6000',step=step,target=6000,at=time.time()))
            if step>=6000:
                if not (TRAIN/'STOP').exists():
                    save(TRAIN/'STOP',dict(owner='user-feature-whisper6000-evaluation32',at=time.time(),target_step=6000,
                         reason='User requested both evaluations on four nodes after complete6000 save; preserve full resume state'))
                break
            time.sleep(.1)
        while not (TRAIN/'step_006000.complete.json').exists():
            if (WORK/'STOP').exists():return
            time.sleep(1)
        # Native complete.json is published only after the entire state is saved.
        # Full SHA, tensor and identity checks occur in the evaluation preparer.
        save(WORK/'CHECKPOINT_PUBLISHED.json',dict(at=time.time(),marker=read(TRAIN/'step_006000.complete.json'),
              checkpoint=str(TRAIN/'step_006000.pt'),observed_training_step=read(TRAIN/'status.json')['step']))
        while not (WORK/'CODE_READY.json').exists():
            if (WORK/'STOP').exists():return
            save(WORK/'GATE_STATUS.json',dict(stage='checkpoint_saved_waiting_evaluation_code',at=time.time()))
            time.sleep(2)
        save(WORK/'GATE_STATUS.json',dict(stage='launching_evaluation32',at=time.time()))
        cli=ROOT/'scripts/p2_current_dual32_eval_20261108.py'
        result=subprocess.run([sys.executable,'-u',str(cli),'run','--work',str(WORK)],cwd=ROOT)
        if result.returncode:raise RuntimeError('Evaluation controller exited with '+str(result.returncode))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(WORK/'GATE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
