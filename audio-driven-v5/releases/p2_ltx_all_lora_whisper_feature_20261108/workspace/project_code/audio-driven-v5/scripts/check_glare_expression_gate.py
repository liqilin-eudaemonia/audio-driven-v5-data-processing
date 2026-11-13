#!/usr/bin/env python3
"""Keep frown and surprise candidates that stand out from the same clip.

The vendor score is not comparable across clips and saturates on a resting brow
or an open mouth. This gate only drops existing intervals. It does not add any.
A frown covering the whole clip is kept when the brow stays low and the detector
score stays high on that same interval. Original detector files are left unchanged.
"""
import argparse
import json
from pathlib import Path
import numpy as np

PARAMS = {
    'smooth_frames': 5,
    'frown_min_brow_down': 0.20,
    'frown_min_lift': 0.09,
    'frown_max_coverage': 0.90,
    'frown_sustained_min_brow': 0.50,
    'frown_sustained_min_score': 0.64,
    'surprise_min_brow': 0.35,
    'surprise_min_lift': 0.40,
    'surprise_max_jaw': 0.15,
    'surprise_max_coverage': 0.85,
}


def smooth(values, window):
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.pad(values, (pad, pad), mode='edge'), kernel, mode='valid')


def blendshapes(pkl_path):
    import pickle
    with Path(pkl_path).open('rb') as handle:
        data = pickle.load(handle)
    n = len(data)
    signal = {name: np.zeros(n, np.float32) for name in ('brow_down', 'brow_inner', 'brow_outer', 'jaw')}

    def pair(blend, left, right):
        return 0.5 * (float(blend.get(left) or 0) + float(blend.get(right) or 0))

    for index, item in enumerate(data):
        if not isinstance(item, dict) or item.get('detected') is not True:
            continue
        blend = item.get('blendshapes') or {}
        signal['brow_down'][index] = pair(blend, 'browDownLeft', 'browDownRight')
        signal['brow_inner'][index] = float(blend.get('browInnerUp') or 0)
        signal['brow_outer'][index] = pair(blend, 'browOuterUpLeft', 'browOuterUpRight')
        signal['jaw'][index] = float(blend.get('jawOpen') or 0)
    return signal


def _contrast(values, start, end):
    inside = np.asarray(values[start:end], dtype=np.float32)
    outside = np.concatenate((values[:start], values[end:]))
    inside_median = float(np.median(inside)) if inside.size else 0.0
    outside_median = float(np.median(outside)) if outside.size else None
    lift = None if outside_median is None else inside_median - outside_median
    return inside_median, outside_median, lift, (end - start) / max(1, len(values))


def assess_frown(brow_down, scores, start, end, params=PARAMS):
    level, baseline, lift, coverage = _contrast(smooth(brow_down, params['smooth_frames']), start, end)
    inside_scores = np.asarray(scores[start:end], dtype=np.float32)
    score_mean = float(np.mean(inside_scores)) if inside_scores.size else 0.0
    sustained = (coverage >= params['frown_max_coverage']
                 and level >= params['frown_sustained_min_brow']
                 and score_mean >= params['frown_sustained_min_score'])
    checks = {
        'brow_down': level >= params['frown_min_brow_down'] or sustained,
        'raised_above_clip': (lift is not None and lift >= params['frown_min_lift']) or sustained,
        'not_whole_clip': coverage < params['frown_max_coverage'] or sustained,
    }
    retained = all(checks.values())
    return {'retain': retained, 'failed_checks': [name for name, ok in checks.items() if not ok],
            'metrics': {'brow_down': level, 'baseline': baseline, 'lift': lift, 'coverage': coverage,
                        'score_mean': score_mean,
                        'path': 'sustained' if sustained else ('contrast' if retained else None)}}


def assess_surprise(brow_inner, brow_outer, jaw, start, end, params=PARAMS):
    window = params['smooth_frames']
    inner_level, _, inner_lift, coverage = _contrast(smooth(brow_inner, window), start, end)
    outer_level, _, outer_lift, _ = _contrast(smooth(brow_outer, window), start, end)
    jaw_level = float(np.median(smooth(jaw, window)[start:end])) if end > start else 0.0
    brow_level = max(inner_level, outer_level)
    lift = max(value for value in (inner_lift, outer_lift) if value is not None) if inner_lift is not None else None
    checks = {
        'brows_up': brow_level >= params['surprise_min_brow'],
        'raised_above_clip': lift is not None and lift >= params['surprise_min_lift'],
        'mouth_not_open': jaw_level < params['surprise_max_jaw'],
        'not_whole_clip': coverage < params['surprise_max_coverage'],
    }
    return {'retain': all(checks.values()), 'failed_checks': [name for name, ok in checks.items() if not ok],
            'metrics': {'brow_inner': inner_level, 'brow_outer': outer_level, 'brow_lift': lift,
                        'jaw': jaw_level, 'coverage': coverage}}


def gate_events(signal, events, kind, scores=None):
    kept = []
    decisions = []
    if scores is None:
        scores = np.zeros(len(signal['brow_down']), dtype=np.float32)
    for event in events:
        start, end = event['start_frame'], event['end_frame_exclusive']
        if kind == 'frowning':
            decision = assess_frown(signal['brow_down'], scores, start, end)
        else:
            decision = assess_surprise(signal['brow_inner'], signal['brow_outer'], signal['jaw'], start, end)
        decision = {**decision, 'type': kind, 'start': event['start'], 'end': event['end'],
                    'peak_score': event.get('peak_score')}
        decisions.append(decision)
        if decision['retain']:
            kept.append(event)
    return kept, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Directory with samples.json and results/{id}.json')
    parser.add_argument('--landmarks', type=Path, required=True, help='Directory of MediaPipe {id}.pkl files')
    parser.add_argument('--output', type=Path, help='Defaults to <base>/expression_gate_v3')
    args = parser.parse_args()
    base = args.base.resolve()
    output = (args.output or base / 'expression_gate_v3').resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids = json.loads((base / 'samples.json').read_text())['ids']
    selected = []
    diagnostics = []
    for sample_id in ids:
        result = json.loads((base / 'results' / f'{sample_id}.json').read_text())
        channels = result.get('channels') or {}
        signal = blendshapes(args.landmarks / f'{sample_id}.pkl')
        for kind in ('frowning', 'surprised'):
            channel = channels.get(kind) or {}
            events = channel.get('events') or []
            _, decisions = gate_events(signal, events, kind, channel.get('scores'))
            for decision in decisions:
                decision['sample_id'] = sample_id
                diagnostics.append(decision)
                if decision['retain']:
                    selected.append({'sample_id': sample_id, 'type': kind, 'start': decision['start'],
                                     'end': decision['end'], 'peak_score': decision['peak_score'],
                                     'metrics': decision['metrics']})
    summary = {'samples': len(ids), 'original': {kind: sum(item['type'] == kind for item in diagnostics)
                                                 for kind in ('frowning', 'surprised')},
               'retained': {kind: sum(item['type'] == kind for item in selected)
                            for kind in ('frowning', 'surprised')},
               'parameters': PARAMS, 'training_admitted': False,
               'rule': 'existing intervals only; within-clip brow contrast, or a whole-clip frown whose brow median is at least 0.50 and whose interval score mean is at least 0.64'}
    for name, payload in (('protocol.json', summary), ('diagnostics.json', diagnostics), ('selected.json', selected)):
        path = output / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
