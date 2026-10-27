#!/usr/bin/env python3
"""Read-only verification of the first automatic checkpoint while E1 continues."""
import json
import math
import sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / 'src'))
from p2_listen.training import checked_checkpoint, sha_tensor
from p2_listen.contracts import read, digest, atomic_json

torch.set_num_threads(2)
run = OUT / 'runs/20261023/E1'
path = run / 'step_000100.pt'
saved = checked_checkpoint(path)
assert saved['step'] == 100
assert saved['identity'] == read(run / 'run.json')
assert saved['identity']['world'] == 24 and saved['identity']['budget'] == 1268
assert len(saved['identity']['labels']) == 30415
assert saved['identity']['s0'] is False and saved['identity']['feature_loss_enabled'] is False
assert len(saved['rank_rng']) == 24
assert all(x['cpu'] is not None and x['cuda'] is not None for x in saved['rank_rng'])
groups = [g['name'] for g in saved['optimizer']['param_groups']]
assert groups == ['connector', 'events', 'text_lora'], groups
assert not any(k.startswith(('extractor.video_decoder.', 'extractor.audio_decoder.')) for k in saved['model'])
assert all(torch.isfinite(v).all() for v in saved['model'].values())
assert all(torch.isfinite(v).all() for state in saved['optimizer']['state'].values()
           for v in state.values() if isinstance(v, torch.Tensor))
assert all(float(state['step']) == 100 for state in saved['optimizer']['state'].values())
assert digest({k: sha_tensor(v) for k, v in saved['model'].items()}) != saved['identity']['initialization']
rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
rows = [row for row in rows if row['step'] <= 100]
assert [row['step'] for row in rows] == list(range(1, 101))
assert all(math.isfinite(v) for row in rows for v in row['scalars'].values())
assert all(row['scalars']['progress/global_batch'] == 24 for row in rows)
ea = EventAccumulator(str(run / 'tensorboard')); ea.Reload()
tags = ea.Tags()['scalars']
assert len(tags) == 43
assert all(any(v.step == 100 for v in ea.Scalars(tag)) for tag in tags)
assert all(ea.Scalars(tag)[0].step == 1 for tag in tags)
record = dict(checkpoint=str(path), step=100, checkpoint_sha256=read(path.with_suffix('.complete.json'))['sha256'],
              bytes=path.stat().st_size, identity_matches_run=True, world=24, budget=1268,
              rng_states=24, optimizer_groups=groups, optimizer_steps_all_100=True,
              model_and_optimizer_tensors_finite=True, model_changed_from_initialization=True,
              feature_decoders_absent=True, model_tensor_elements=sum(v.numel() for v in saved['model'].values()),
              text_lora_tensor_elements=sum(v.numel() for k, v in saved['model'].items() if k.startswith('text_lora.')),
              tensorboard_scalar_tags=43, tensorboard_step_100_verified=True, global_metrics_steps_verified=100,
              verification_only=True, live_training_interrupted=False)
atomic_json(OUT / 'checkpoint_000100_validation.json', record, immutable=True)
print(json.dumps(record, indent=2))
