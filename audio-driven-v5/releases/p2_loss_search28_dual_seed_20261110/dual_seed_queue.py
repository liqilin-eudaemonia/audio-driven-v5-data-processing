"""Finish existing seed137 without interruption, then score all seven at238."""
from pathlib import Path
import copy
import fcntl
import os
import sys
import time
import traceback

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
from p2_loss_search28.config import CANDIDATES,WORK as TRAIN


def work_for(name,seed):
    return WORK/'evaluations'/(f'{name}_500'if seed==137 else f'{name}_500_seed{seed}')


def validate_frozen_plan():
    plan=read(WORK/'DUAL_SEED_PLAN.json')
    if plan['seeds']!=[137,238]or plan['samples_per_seed']!=64 or plan['candidates']!=list(CANDIDATES):
        raise ValueError('Double-seed experiment changed')
    if sha(WORK/'tuning64.json')!=plan['membership_sha256']:raise ValueError('Membership changed')
    for f,h in plan['source'].items():
        if sha(f)!=h:raise ValueError('Double-seed source changed')
    return plan


def prepare_second(name):
    """Keep the same model/conditioning and change only the generation draw."""
    from p2_combined_whisper_eval import link
    from stage_driver import verify_scoring
    validate_frozen_plan()
    source=work_for(name,137);dest=work_for(name,238)
    verify_scoring(source);old=read(source/'protocol.json')
    if old['seed']!=137 or old['total_samples']!=64:raise ValueError('Wrong first draw')
    dest.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(old);p.update(seed=238,cluster=str(dest/'cluster.json'),metrics_cluster=str(dest/'cluster.json'),
        shared_items_manifest=str(dest/'shared_items.json'))
    if (dest/'protocol.json').exists():
        if read(dest/'protocol.json')!=p:raise ValueError('Second-draw contract changed')
        return dest
    # Export is immutable and already validated. No second model export or conditions computation.
    for folder in ['export','conditions','whisper','negative']:link(source/folder,dest/folder)
    for f in ['REQUEST.json','cluster.json','DATA_CONFIG.json','shared_items.json']:
        save(dest/f,read(source/f),immutable=True)
    if sha(dest/'shared_items.json')!=p['shared_items_sha256']:
        raise ValueError('Copied sample manifest changed')
    for rank in range(4):save(dest/'shards'/f'{rank:02d}.json',read(source/'shards'/f'{rank:02d}.json'),immutable=True)
    save(dest/'protocol.json',p,immutable=True)
    save(dest/'DRAW_IDENTITY.json',dict(generation_seed=238,first_draw=str(source),
         first_draw_protocol_sha256=sha(source/'protocol.json'),checkpoint_sha256=p['arms'][0]['original_sha256'],
         same_conditions=True,same_samples=True,same_detector_seed=True),immutable=True)
    return dest


def run():
    from gpu7_score import score_one
    from search_controller import alive
    from stage_driver import verify_scoring
    validate_frozen_plan()
    stopped=lambda:(WORK/'STOP').exists()or(WORK/'DUAL_SEED_STOP').exists()or(TRAIN/'STOP').exists()
    with (WORK/'dual_seed.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        save(WORK/'DUAL_SEED_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,at=time.time()))
        while not all((work_for(n,137)/'SCORING_COMPLETE.json').exists()for n in CANDIDATES):
            if stopped():return
            controller=read(WORK/'STAGE_CONTROLLER.json')
            if not alive(controller):raise RuntimeError('Seed137 controller stopped before completion; inspect its failure')
            save(WORK/'DUAL_SEED_STATUS.json',dict(stage='waiting_for_existing_seed137',at=time.time(),
                completed=sum((work_for(n,137)/'SCORING_COMPLETE.json').exists()for n in CANDIDATES)))
            time.sleep(30)
        for name in CANDIDATES:verify_scoring(work_for(name,137))
        with(WORK/'stage_driver.lock').open('a')as stage_lock:
            while True:
                if stopped():return
                try:fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            save(WORK/'STAGE_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,target=500,
                 device='node11GPU7',seed=238,at=time.time()))
            for name in CANDIDATES:
                if stopped():return
                validate_frozen_plan();work=prepare_second(name)
                if(work/'STOP').exists():return
                save(WORK/'DUAL_SEED_STATUS.json',dict(stage='scoring',candidate=name,seed=238,at=time.time()))
                score_one(work)
            from dual_seed_compare import generate_report
            generate_report()
            save(WORK/'DUAL_SEED_STATUS.json',dict(stage='comparison_ready_for_review',at=time.time(),
                 candidates=list(CANDIDATES),seeds=[137,238],samples_per_seed=64,automatic_promotion=False))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        save(WORK/'DUAL_SEED_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
