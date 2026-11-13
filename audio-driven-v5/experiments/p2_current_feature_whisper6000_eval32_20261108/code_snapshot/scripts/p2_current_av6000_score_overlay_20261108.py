"""Finish AV6000 scoring on shared node11 GPU7 without stopping other jobs."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read, save, sha
WORK = ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'
RESERVE_MIB = 16*1024
NEW_JOB_MIB = 8*1024


def free_gpu7(report):
    for line in report.splitlines():
        index, used, total, util = [float(v.strip()) for v in line.split(',')]
        if int(index) == 7:
            return dict(used_mib=used, total_mib=total, free_mib=total-used, util=util)
    raise ValueError('Physical GPU7 missing from resource report')


def launch_budget(free_mib, active):
    return max(0, min(8-active, int((free_mib-RESERVE_MIB)//NEW_JOB_MIB)))


def queue(work):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:
        raise ValueError('Only node11 may run this controller')
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = read(work/'protocol.json')
        cluster = read(p['metrics_cluster'])
        assert [(n['host'], n['gpus']) for n in cluster['nodes']] == [('172.29.189.119', [7])]
        assert read(work/'GENERATION_COMPLETE.json')['total_outputs'] == 400
        assert read(work/'METRICS_SMOKE_COMPLETE.json')['status'] == 'passed'
        assert not (work/'STOP').exists()
        if (work/'HANDOFF_COMPLETE.json').exists():
            return
        from p2_testsets_eval.runtime import ready
        arm = p['arms'][0]['name']
        assert all(ready(work/'generated'/arm/x['key']) for x in read(work/'shared_items.json'))
        out = work/'overlay_scoring'
        out.mkdir(exist_ok=True)
        record = dict(pid=os.getpid(), at=time.time(), physical_gpu=7,
                      start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21])
        save(work/'CONTROLLER.json', record)
        save(out/'CONTROLLER.json', record)
        save(out/'EXECUTION.json', dict(at=time.time(), source_sha256=sha(SELF),
             max_workers=8, reserve_mib=RESERVE_MIB, memory_budget_per_new_worker_mib=NEW_JOB_MIB,
             latest_user_authorized_overlay=True, original_tasks_untouched=True,
             original_generation_and_metric_algorithms_unchanged=True))
        import listen_cluster
        from p2_testsets_eval import metrics
        native_transport = listen_cluster.transport
        latest = {}
        def transport(cfg, node, command):
            assert node['host'] == '172.29.189.119'
            return native_transport(cfg, node, ['env', 'CUDA_VISIBLE_DEVICES=7', *command])
        def available(report):
            latest.clear()
            latest.update(free_gpu7(report))
            save(out/'RESOURCE.json', dict(at=time.time(), **latest))
            return [7] if launch_budget(latest['free_mib'], 0) else []
        def slots(reports, active):
            count = sum(v[2] == 7 for v in active.values())
            budget = launch_budget(latest.get('free_mib', 0), count)
            eligible = [(name, gpu) for name, capacity, gpus in reports for gpu in gpus if gpu == 7]
            save(out/'SLOT_BUDGET.json', dict(at=time.time(), active=count,
                  new_worker_budget=budget, reserve_mib=RESERVE_MIB,
                  free_mib=latest.get('free_mib'), max_workers=8))
            return [eligible[0]]*budget if eligible else []
        save(work/'status.json', dict(stage='metrics_overlay', at=time.time(), physical_gpu=7))
        with patch.object(listen_cluster, 'transport', transport), \
             patch.object(metrics, 'CLI', ROOT/'scripts/p2_pipelined_testsets_eval.py'), \
             patch.object(metrics, 'idle_gpus', available), \
             patch.object(metrics, 'balanced_metric_slots', slots):
            metrics.queue(work)
        if (work/'STOP').exists():
            save(work/'status.json', dict(stage='stopped', at=time.time()))
            return
        import p2_current_av6000_sft2752_eval_20261107 as original
        original.final(work)
        save(work/'HANDOFF_COMPLETE.json', dict(outputs=400, step=6000, planner_step=2752,
             overlay_scoring=True, physical_gpu=7, at=time.time()))
        save(work/'status.json', dict(stage='complete', outputs=400, at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    args = parser.parse_args()
    try:
        queue(args.work.resolve())
    except Exception as exc:
        save(args.work/'overlay_scoring/FAILURE.json',
             dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
