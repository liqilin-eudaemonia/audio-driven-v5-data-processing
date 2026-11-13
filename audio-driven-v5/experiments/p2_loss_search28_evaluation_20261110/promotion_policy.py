"""Conservative, predeclared generation-based successive-budget decisions.

This is an engineering screening policy, not a claim of globally optimal loss
weights or a calibrated test of contextual appropriateness. All candidates use
the same clips, reference labels and paired source-group bootstrap draws.
"""
from pathlib import Path
import json
import math
import sys
import time
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(WORK)]
from p2_listen.contracts import read, sha, atomic_json
from p2_event_presence.contracts import CLASSES, KNOWN, digest, validate_record, rows
from p2_loss_search28.config import WORK as TRAIN, CANDIDATES

DEFAULT = dict(
    schema='losssearch-generation-screening-v1', bootstrap_draws=20000, seed=137,
    family_alpha=.05, minimum_source_groups=12, minimum_common_class_coverage=.90,
    minimum_retained_reference_positive_fraction=.90, minimum_valid_bootstrap_fraction=.99,
    reaction_min_gain_by_stage={'500': .05, '1500': .02, '3000': .02},
    noninferiority_margins=dict(visual_f1=.02, audio_f1=.02, lpips=.01, psnr=.25,
                               ssim=.01, false_sound_rate=.02, missed_response_rate=.02,
                               supported_class_f1=.05),
    keep_baseline=True, require_repeat_dominance_after_first_stage=True,
    budget_targets={'1500': 3, '3000': 2, '6000': 2},
    uncertainty_action='retain; no forced top-k or training-loss tie-break',
    reference_policy='frozen_reference_positive_support; common known labels only; unknown never negative',
    interpretation='Automatic pilot detector category agreement; not contextual appropriateness, identity, sound quality or AV synchronization',
    final_policy='6000 produces finalists, not an automatic best-model claim; compare extra generation seeds, incumbent and independent holdout',
    statistical_scope='approximate paired source-group percentile bootstrap; per-stage ordered-pair multiplicity adjustment; not cross-stage familywise proof',
)


def freeze_policy():
    # Invoke before any candidate outcome is available; immutable on subsequent runs.
    atomic_json(WORK / 'PROMOTION_POLICY.json', DEFAULT, immutable=True)
    return DEFAULT


def validate_policy(policy):
    if policy != DEFAULT:
        raise ValueError('Promotion policy differs from predeclared policy')


def indexed(values, ids, what):
    result = {r['id']: r for r in values}
    if len(result) != len(values) or set(result) != set(ids):
        raise ValueError('Wrong/duplicate population: ' + what)
    return result


def arrays(candidates, membership, policy):
    """Build raw corpus confusion counts and paired per-clip quality measurements."""
    ids = sorted(r['id'] for r in membership)
    groups = {r['id']: r['group_id'] for r in membership}
    if len(ids) != len(set(ids)) or not all(groups.values()):
        raise ValueError('Validation IDs/source groups missing or duplicate')
    names = list(candidates)
    if not names: raise ValueError('No candidates')
    indexed_data = {}
    for name, value in candidates.items():
        indexed_data[name] = {key: indexed(value[key], ids, name + '/' + key)
                              for key in ('events', 'video', 'audio')}
        for row in value['events']: validate_record(row)
    reference = indexed_data[names[0]]['events']
    protocols = {r['protocol_identity'] for v in candidates.values() for r in v['events']}
    if len(protocols) != 1: raise ValueError('Mixed event detector protocols')
    for name in names:
        for sid in ids:
            a, b = reference[sid], indexed_data[name]['events'][sid]
            if a['reference_identity'] != b['reference_identity'] or a['reference'] != b['reference']:
                raise ValueError('Reference media/labels changed across candidates')
    supported, masks, issues = {}, {}, []
    for mod, classes in CLASSES.items():
        supported[mod] = [c for c in classes if any(reference[s]['reference'][mod][c]['state'] == 'present' for s in ids)]
        if not supported[mod]: issues.append('No reference-positive classes: ' + mod)
        for c in classes:
            mask = np.array([reference[s]['reference'][mod][c]['state'] in KNOWN and
                all(indexed_data[n]['events'][s]['generated'][mod][c]['state'] in KNOWN for n in names) for s in ids])
            masks[mod, c] = mask
            if mask.mean() < policy['minimum_common_class_coverage']:
                issues.append(f'Insufficient common evaluable coverage: {mod}/{c}')
            positive = np.array([reference[s]['reference'][mod][c]['state'] == 'present' for s in ids])
            if positive.any() and (mask & positive).sum() / positive.sum() < policy['minimum_retained_reference_positive_fraction']:
                issues.append(f'Insufficient retained GT positives: {mod}/{c}')
    group_names = sorted(set(groups.values()))
    if len(group_names) < policy['minimum_source_groups']: issues.append('Too few independent source groups')
    group_index = np.array([group_names.index(groups[s]) for s in ids])
    output = {}
    for name in names:
        data = indexed_data[name]; counts = {}; metrics = {}
        for mod, cs in supported.items():
            counts[mod] = np.zeros((len(ids), len(cs), 3), dtype=float)
            for j, c in enumerate(cs):
                gt = np.array([reference[s]['reference'][mod][c]['state'] == 'present' for s in ids])
                pred = np.array([data['events'][s]['generated'][mod][c]['state'] == 'present' for s in ids])
                mask = masks[mod, c]
                counts[mod][:, j, :] = np.stack([gt & pred & mask, ~gt & pred & mask, gt & ~pred & mask], axis=1)
        for key in ('lpips', 'psnr', 'ssim'):
            values = [data['video'][s].get('metrics', {}).get(key) for s in ids]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                issues.append('Missing/nonfinite video metric: ' + name + '/' + key)
                values = [float('nan')] * len(ids)
            metrics[key] = (np.array(values, dtype=float), np.ones(len(ids)))
        false, missed, silent, active = [], [], [], []
        for sid in ids:
            row = data['audio'][sid]; a = row.get('audio', {}).get('comparison', {}).get('acoustic', {})
            keys = ('any_reference_activity', 'false_activity_on_silent_clip', 'missed_entire_response')
            if row.get('errors') or any(type(a.get(k)) is not bool for k in keys):
                issues.append('Invalid audio activity metrics: ' + name + '/' + sid)
                false.append(float('nan')); missed.append(float('nan')); silent.append(0); active.append(0)
                continue
            # Reference activity must be identical for every candidate.
            other = indexed_data[names[0]]['audio'][sid].get('audio', {}).get('comparison', {}).get('acoustic', {})
            if a['any_reference_activity'] != other.get('any_reference_activity'):
                raise ValueError('Reference audio activity changed')
            active.append(int(a['any_reference_activity'])); silent.append(int(not a['any_reference_activity']))
            false.append(int(a['false_activity_on_silent_clip'])); missed.append(int(a['missed_entire_response']))
        metrics['false_sound_rate'] = (np.array(false), np.array(silent))
        metrics['missed_response_rate'] = (np.array(missed), np.array(active))
        output[name] = dict(counts=counts, metrics=metrics)
    return output, dict(ids=ids, group_index=group_index, source_groups=group_names,
                        supported=supported, issues=sorted(set(issues)),
                        common_coverage={mod + '/' + c: float(mask.mean()) for (mod, c), mask in masks.items()})


def bootstrap_metrics(value, info, weights):
    """Identical group resampling for all candidates; no per-clip F1 averaging."""
    gi = info['group_index']; groups = len(info['source_groups']); result = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for mod, counts in value['counts'].items():
            grouped = np.zeros((groups, *counts.shape[1:])); np.add.at(grouped, gi, counts)
            sums = (weights @ grouped.reshape(groups, -1)).reshape(len(weights), *counts.shape[1:])
            f1 = 2 * sums[:, :, 0] / (2 * sums[:, :, 0] + sums[:, :, 1] + sums[:, :, 2])
            result[mod + '_f1'] = f1.mean(axis=1) if f1.shape[1] else np.full(len(weights), np.nan)
            for j, c in enumerate(info['supported'][mod]): result['class/' + mod + '/' + c] = f1[:, j]
        result['reaction'] = (result['visual_f1'] + result['audio_f1']) / 2
        for metric, (numerator, denominator) in value['metrics'].items():
            grouped = np.zeros((groups, 2)); np.add.at(grouped, gi, np.stack([numerator, denominator], axis=1))
            sums = weights @ grouped
            result[metric] = sums[:, 0] / sums[:, 1]
    return result


def decide(candidates, membership, policy, step, previous=None):
    validate_policy(policy)
    if step not in (500, 1500, 3000): raise ValueError('No automatic best-model decision at6000')
    data, info = arrays(candidates, membership, policy)
    names = list(candidates)
    if 'baseline' not in names: raise ValueError('Matched baseline required')
    rng = np.random.default_rng(policy['seed'])
    g = len(info['source_groups'])
    weights = rng.multinomial(g, np.full(g, 1/g), size=policy['bootstrap_draws']).astype(float)
    weights = np.vstack([np.ones((1, g)), weights])
    measures = {n: bootstrap_metrics(v, info, weights) for n, v in data.items()}
    # Per-stage ordered-pair correction; multi-stage selection remains adaptive.
    tail = policy['family_alpha'] / max(1, len(names) * (len(names)-1)) / 2
    comparisons = []; dominance = []
    for better in names:
        for worse in names:
            if better == worse: continue
            bounds = {}; credible = not info['issues']
            for metric, a in measures[better].items():
                b = measures[worse][metric]
                direction = -1 if metric in ('lpips', 'false_sound_rate', 'missed_response_rate') else 1
                delta = direction * (a - b); known = np.isfinite(delta[1:])
                if not np.isfinite(delta[0]) or known.mean() < policy['minimum_valid_bootstrap_fraction']:
                    credible = False
                    bounds[metric] = dict(valid_fraction=float(known.mean()), lower=None, upper=None, delta=None)
                    continue
                lower, upper = np.quantile(delta[1:][known], [tail, 1-tail])
                bounds[metric] = dict(valid_fraction=float(known.mean()), lower=float(lower), upper=float(upper), delta=float(delta[0]))
                if metric == 'reaction': threshold = policy['reaction_min_gain_by_stage'][str(step)]
                else: threshold = -policy['noninferiority_margins']['supported_class_f1' if metric.startswith('class/') else metric]
                if lower < threshold: credible = False
            repeated = step == 500 or (previous is not None and [better, worse] in previous.get('dominance_current', []))
            if credible: dominance.append([better, worse])
            comparisons.append(dict(better=better, worse=worse, bounds=bounds,
                                    credible_current=credible, repeated=repeated,
                                    eligible_to_eliminate=credible and repeated and worse != 'baseline'))
    eliminated = sorted({r['worse'] for r in comparisons if r['eligible_to_eliminate']})
    kept = [n for n in names if n not in eliminated]
    target = {500:1500, 1500:3000, 3000:6000}[step]
    budget = policy['budget_targets'][str(target)]
    point = {n: {k: float(v[0]) if np.isfinite(v[0]) else None for k, v in ms.items()} for n, ms in measures.items()}
    return dict(schema=policy['schema'], completed_step=step, target=target, candidates=kept,
                eliminated=eliminated, dominance_current=dominance, comparisons=comparisons,
                point_metrics=point, common_coverage=info['common_coverage'], issues=info['issues'],
                source_groups=g, samples=len(membership), bootstrap_draws=policy['bootstrap_draws'],
                per_tail_probability=tail, budget_target=budget, exceeds_budget_target=len(kept)>budget,
                extra_update_steps_vs_target=max(0,len(kept)-budget)*(target-step),
                decision_scope='conservative pilot screening, not a final best-model claim',
                uncertainty_action=policy['uncertainty_action'])


def load_stage(step, names):
    from stage_driver import verify_scoring
    manifest = WORK / f'tuning{64 if step == 500 else 200}.json'
    if sha(manifest) != read(WORK/'VALIDATION_MEMBERSHIP_ACCEPTED.json')['outputs'][manifest.name]:
        raise ValueError('Validation population changed after freezing')
    membership = read(manifest)
    source = {}; candidates = {}
    for name in names:
        work = WORK / 'evaluations' / f'{name}_{step}'
        receipt = verify_scoring(work)
        p = read(work / 'protocol.json'); arm = p['arms'][0]
        if receipt.get('success') is not True or arm['step'] != step or p['checkpoint_identity']['experiment']['loss_search']['candidate'] != name:
            raise ValueError('Wrong scoring candidate/checkpoint')
        if p['total_samples'] != len(membership) or p['seed'] != 137:
            raise ValueError('Wrong generation population or seed')
        marker = read(TRAIN / 'runs' / name / f'step_{step:06d}.complete.json')
        if receipt['checkpoint_sha256'] != marker['sha256']: raise ValueError('Checkpoint changed')
        base = 'metrics/' + arm['name'] + '/seamless/'
        files = dict(events='event_pairs_fixed_reference.jsonl', video=base+'pairs.jsonl', audio=base+'audio_pairs.jsonl')
        if not set(files.values()) <= set(receipt['evidence']): raise ValueError('Per-clip score evidence not sealed')
        for f, h in receipt['evidence'].items():
            if sha(work/f) != h: raise ValueError('Scoring evidence changed: ' + f)
        candidates[name] = {key: rows(work/f) for key, f in files.items()}
        source[str(work/'SCORING_COMPLETE.json')] = sha(work/'SCORING_COMPLETE.json')
    return candidates, membership, source


def promote(step):
    from stage_driver import stage_candidates
    policy = read(WORK / 'PROMOTION_POLICY.json'); validate_policy(policy)
    names = stage_candidates(step)
    data, membership, evidence = load_stage(step, names)
    path = WORK / f'DECISION_{step}.json'
    policy_sha=sha(WORK/'PROMOTION_POLICY.json');code_sha=sha(Path(__file__))
    if path.exists():
        decision=read(path)
        if decision.get('policy_sha256')!=policy_sha or decision.get('decision_code_sha256')!=code_sha or decision.get('evidence')!=evidence or decision.get('completed_step')!=step:
            raise ValueError('Frozen promotion inputs/source changed')
        if decision.get('target')!={500:1500,1500:3000,3000:6000}[step]or not decision.get('candidates')or any(n not in names for n in decision['candidates']):
            raise ValueError('Invalid frozen promotion')
        # Reuse the sealed decision; avoid numerical/formatting drift on restart.
    else:
        previous = None
        if step != 500:
            previous_step = {1500:500, 3000:1500}[step]
            previous = read(WORK / f'DECISION_{previous_step}.json')
            if previous['candidates'] != names or previous['policy_sha256']!=policy_sha or previous['decision_code_sha256']!=code_sha:
                raise ValueError('Candidate/policy history changed')
        decision = decide(data, membership, policy, step, previous)
        decision.update(policy_sha256=policy_sha,decision_code_sha256=code_sha,evidence=evidence)
        atomic_json(path, decision, immutable=True)
    promotion = dict(target=decision['target'], basis='generated_av_quality', candidates=decision['candidates'],
                     evidence=[str(path), *evidence], decision_sha256=sha(path),
                     evidence_sha256=evidence, policy_sha256=decision['policy_sha256'])
    atomic_json(TRAIN/f"promotion_{decision['target']}.json", promotion, immutable=True)
    return decision


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'promote'])
    p.add_argument('--step', type=int, choices=[500,1500,3000]); args=p.parse_args()
    print(json.dumps(freeze_policy() if args.action=='freeze' else promote(args.step), ensure_ascii=False))
