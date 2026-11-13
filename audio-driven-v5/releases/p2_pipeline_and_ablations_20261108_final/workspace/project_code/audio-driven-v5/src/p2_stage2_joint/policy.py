"""Predeclared short-trial safety/improvement gates; never auto-launch1000 steps."""
from pathlib import Path
import math
from p2_listen.contracts import read,sha,digest,atomic_json

DEFAULT=dict(schema='stage2-trial-policy-v1',validation_size=64,
    planning_ce_max_relative_increase=.10,lpips_max_absolute_increase=.01,
    false_sound_max_absolute_increase=.02,format_error_max_absolute_increase=.02,
    reaction_f1_min_absolute_improvement=.02,
    selection='100steps safety only;300steps safety plus reaction improvement; no automatic1000step launch',
    status='engineering screening thresholds; not statistical significance or detector calibration')


def decide(baseline,current,policy,step):
    required=('planning_ce','lpips','false_sound_rate','format_error_rate')+ (('reaction_f1',)if step==300 else ())
    if step not in (100,300):raise ValueError('Only100/300 trial decisions supported')
    for result in (baseline,current):
        if result.get('samples')!=64 or result.get('complete')is not True:
            return dict(decision='hold',reason='Incomplete64-sample evaluation')
        if any(type(result.get(k))not in (int,float)or not math.isfinite(result[k])for k in required):
            return dict(decision='hold',reason='Missing/unknown/nonfinite required metrics')
    changes={k:current[k]-baseline[k]for k in required}
    failures=[]
    if current['planning_ce']>baseline['planning_ce']*(1+policy['planning_ce_max_relative_increase']):failures.append('planning_ce')
    for metric,key in [('lpips','lpips_max_absolute_increase'),('false_sound_rate','false_sound_max_absolute_increase'),('format_error_rate','format_error_max_absolute_increase')]:
        if changes[metric]>policy[key]:failures.append(metric)
    if failures:return dict(decision='stop',reason='Safety regression',failed_metrics=failures,changes=changes)
    if step==100:return dict(decision='continue_to_300',changes=changes)
    if changes['reaction_f1']>=policy['reaction_f1_min_absolute_improvement']:
        return dict(decision='eligible_for_1000_review',changes=changes,auto_launch=False)
    return dict(decision='hold',reason='No clear reaction improvement; retainStage1',changes=changes)


def require_evaluation(c,step):
    from .evaluation import verify_report
    folder=Path(c['output']);report=verify_report(c,step)
    if (not report['success'] or report['checkpoint_sha256']!=read(folder/f'step_{step:06d}.complete.json')['sha256']
        or report['validation_sha256']!=c['validation_manifest_sha256']or report['policy_sha256']!=c['policy_sha256']):
        raise ValueError('Evaluation barrier identity mismatch')
    if step==0:
        if report['summary']['samples']!=64 or not report['summary']['complete']:raise ValueError('Complete baseline evaluation required')
    elif step==100:
        decision=read(folder/'evaluation'/'DECISION_000100.json')
        if decision['report_sha256']!=sha(folder/'evaluation'/'step_000100'/'REPORT.json')or decision['baseline_report_sha256']!=sha(folder/'evaluation'/'step_000000'/'REPORT.json')or decision['policy_sha256']!=c['policy_sha256']or decision['decision']!='continue_to_300':
            raise ValueError('100step evaluation did not authorize continuation')
    return report
