"""Launch the approved base-H AV+Feature run; verify 3 updates, then resume."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import atomic_json, read
from p2_matrix16.config import load
from p2_j_s1_queue import commands
from listen_cluster import launch


def verify_metrics(rows):
    if [r['step'] for r in rows] != [1, 2, 3]:
        raise ValueError('Expected exactly the first 3 formal updates')
    for r in rows:
        for k in ('av', 'av_video', 'av_audio', 'feature_video', 'feature_audio', 'weighted_total'):
            if not math.isfinite(r[k]):
                raise ValueError('Nonfinite ' + k)
        expected = r['av'] + .025 * (r['feature_video'] + r['feature_audio'])
        if r['event'] != 0 or r['grad_norm']['events'] != 0:
            raise ValueError('Disabled event branch contributed')
        if not math.isclose(r['weighted_total'], expected, rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError('Objective differs from approved weights')
    for group in ('connector', 'features', 'text_lora'):
        if not all(math.isfinite(r['grad_norm'][group]) for r in rows):
            raise ValueError('Nonfinite gradient: ' + group)
        if not any(r['grad_norm'][group] > 0 for r in rows):
            raise ValueError('Missing gradient: ' + group)
    return dict(success=True, verified_steps=3, objective_verified=True,
                active_gradient_groups=['connector', 'features', 'text_lora'],
                event_gradient_zero=True,
                rank0_peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
                initial_mean_step_seconds=sum(r['seconds'] for r in rows) / 3)


def run(config, cluster):
    c = load(config, formal=True)
    if (c['text_condition'], c['loss'], c['stage'], c.get('qwen_hidden_source')) != ('guided', 'av_feature', 1, 'base_pre_sft'):
        raise ValueError('This launcher requires the approved base-H AV+Feature configuration')
    if c['event_head_training'] or c['qwen_lora']['enabled'] or c['training']['global_batch'] != 32:
        raise ValueError('Event/Qwen/batch configuration differs')
    if [c['loss_weights'][k] for k in ('av', 'video_feature', 'audio_feature')] != [1, .025, .025]:
        raise ValueError('Loss weights differ')
    out = Path(c['output']); q = out / 'queue'; q.mkdir(exist_ok=True)

    def status(state, **extra):
        value = dict(state=state, time=time.time(), config=str(config), world=32,
                     budget=c['training']['steps'], **extra)
        atomic_json(q / 'state.json', value)
        print(json.dumps(value), flush=True)

    def render(**kwargs):
        result = commands(cluster, config, **kwargs)
        for cmd in result:
            idx = cmd.index(str(ROOT / 'scripts/p2_clip_events.py'))
            cmd[idx] = str(ROOT / 'scripts/p2_j_duration_train.py')
        return result

    with (q / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (out / 'run.json').exists():
                raise ValueError('Existing run: use its complete checkpoint for explicit recovery')
            checkpoint = out / 'step_000003.pt'
            initial = render(stop_after=3)
            continuation = render(resume=checkpoint)
            atomic_json(out / 'launch_commands.json', dict(cluster=cluster, initial=initial, continuation=continuation))
            status('initial_32gpu_training', validation_steps=3)
            launch(cluster, initial, q / 'initial3')
            import torch
            from p2_matrix16.checkpoint import checked
            torch.set_num_threads(2)
            status('checking_initial_checkpoint')
            saved = checked(checkpoint)
            if saved['step'] != 3 or saved['identity']['world'] != 32 or saved['identity']['experiment'] != c:
                raise ValueError('Initial checkpoint identity differs')
            del saved
            rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
            result = verify_metrics(rows)
            atomic_json(out / 'GPU_VALIDATION.json', dict(checkpoint=str(checkpoint), **result))
            ready = read(out / 'READY.json')
            ready.update(training_started=True, gpu_training_validated=True, world=32)
            atomic_json(out / 'READY.json', ready)
            if (out / 'STOP').exists():
                status('stopped_after_validation'); return
            status('training', resume_step=3, **result)
            launch(cluster, continuation, q / 'continue_epoch')
            final = read(out / 'status.json')
            if not final['complete'] or final['step'] != c['training']['steps']:
                raise ValueError('Training ended before budget')
            status('complete', step=final['step'])
        except Exception as exc:
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cluster', type=Path, required=True)
    args = p.parse_args()
    run(args.config.resolve(), read(args.cluster))
