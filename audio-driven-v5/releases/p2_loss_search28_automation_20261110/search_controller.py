"""Continue training budgets only after sealed generation-based stage decisions."""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json
from p2_loss_search28.config import WORK as TRAIN,source_identity


def alive(record):
    p=Path('/proc')/str(record['pid'])
    if not p.exists():return False
    fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return fields[19]==record['start_ticks'] and fields[0]not in ('Z','X')


def run():
    from promotion_policy import promote,validate_policy,load_stage
    from stage_driver import evaluate,stage_candidates
    from p2_loss_search28.queue import run as train_queue
    stopped=lambda:(WORK/'STOP').exists()or(TRAIN/'STOP').exists()
    validate_policy(read(WORK/'PROMOTION_POLICY.json'))
    if read(TRAIN/'PREFLIGHT_ACCEPTED.json')['source']!=source_identity():
        raise ValueError('Training source changed since acceptance')
    acceptance=read(WORK/'AUTOMATION_CPU_ACCEPTED.json')
    if not acceptance['success']:raise ValueError('Automation tests not accepted')
    for f,h in acceptance['source'].items():
        if sha(f)!=h:raise ValueError('Automation source changed since review')
    with(WORK/'search_controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        atomic_json(WORK/'SEARCH_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,at=time.time()))
        # The already-running stage500 driver owns its own lock and children.
        # Its continued existence is checked; an old status file is not liveness.
        while True:
            if stopped():return
            status=WORK/'STAGE_500_STATUS.json'
            if status.exists()and read(status).get('stage')=='generation_review_ready':break
            controller=WORK/'STAGE_CONTROLLER.json'
            if not controller.exists()or not alive(read(controller)):
                raise RuntimeError('Initial scoring driver is not live; inspect failure before restarting')
            atomic_json(WORK/'SEARCH_STATUS.json',dict(stage='waiting_for_500_training_and_scoring',at=time.time()))
            time.sleep(30)
        for step,target in ((500,1500),(1500,3000),(3000,6000)):
            if stopped():return
            decision=promote(step)
            atomic_json(WORK/'SEARCH_STATUS.json',dict(stage='promoted_training',completed_step=step,
                target=target,candidates=decision['candidates'],exceeds_budget_target=decision['exceeds_budget_target'],
                extra_update_steps_vs_target=decision['extra_update_steps_vs_target'],at=time.time()))
            # Restores each candidate's own model, Adam and28 RNGs. Completed
            # stages are validated and skipped, never repeated from scratch.
            train_queue()
            if stopped():return
            evaluate(target)
            if stopped():return
        names=stage_candidates(6000)
        _,_,evidence=load_stage(6000,names)
        atomic_json(WORK/'FINALISTS_READY.json',dict(success=True,candidates=names,step=6000,
            evidence_sha256=evidence,at=time.time(),best_model_selected=False,
            pending=['extra_generation_seeds','comparison_with_incumbent','independent_holdout_confirmation'],
            interpretation='Completed staged candidates; no automatic global-optimum claim'))
        atomic_json(WORK/'SEARCH_STATUS.json',dict(stage='finalist_confirmation_required',at=time.time(),candidates=names))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        atomic_json(WORK/'SEARCH_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
