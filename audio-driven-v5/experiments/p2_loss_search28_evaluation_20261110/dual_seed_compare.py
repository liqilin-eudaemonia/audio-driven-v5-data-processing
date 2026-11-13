"""Paired two-generation-seed comparison; source groups remain the sampling unit."""
from pathlib import Path
import json
import math
import sys
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json
from p2_event_presence.contracts import rows
import promotion_policy as policy_module

SEEDS=(137,238)


def strict_mean(values):
    return float(np.mean(values)) if all(type(v)in(int,float)and math.isfinite(v)for v in values)else None


def compare(candidates,membership,policy):
    """candidates[name][seed] holds separate event/video/audio populations."""
    policy_module.validate_policy(policy)
    flat={}
    for name,draws in candidates.items():
        if set(draws)!=set(SEEDS):raise ValueError('Both generation seeds required')
        for seed in SEEDS:flat[f'{name}@{seed}']=draws[seed]
    # One common known-label mask across all candidates AND both seeds.
    data,info=policy_module.arrays(flat,membership,policy)
    g=len(info['source_groups']);rng=np.random.default_rng(policy['seed'])
    weights=rng.multinomial(g,np.full(g,1/g),size=policy['bootstrap_draws']).astype(float)
    weights=np.vstack([np.ones((1,g)),weights])
    raw={key:policy_module.bootstrap_metrics(value,info,weights)for key,value in data.items()}
    averaged={name:{metric:(raw[f'{name}@137'][metric]+raw[f'{name}@238'][metric])/2
                    for metric in raw[f'{name}@137']}for name in candidates}
    names=list(candidates);tail=policy['family_alpha']/max(1,len(names)*(len(names)-1))/2
    comparisons=[]
    for better in names:
        for worse in names:
            if better==worse:continue
            clear=not info['issues'];bounds={}
            for metric,av in averaged[better].items():
                direction=-1 if metric in ('lpips','false_sound_rate','missed_response_rate')else 1
                delta=direction*(av-averaged[worse][metric]);valid=np.isfinite(delta[1:])
                if not np.isfinite(delta[0])or valid.mean()<policy['minimum_valid_bootstrap_fraction']:
                    clear=False;bounds[metric]=dict(delta=None,lower=None,upper=None,valid_fraction=float(valid.mean()));continue
                lo,hi=np.quantile(delta[1:][valid],[tail,1-tail])
                threshold=policy['reaction_min_gain_by_stage']['500'] if metric=='reaction'else -policy['noninferiority_margins']['supported_class_f1'if metric.startswith('class/')else metric]
                if lo<threshold:clear=False
                bounds[metric]=dict(delta=float(delta[0]),lower=float(lo),upper=float(hi),valid_fraction=float(valid.mean()))
            seed_deltas={str(s):float(raw[f'{better}@{s}']['reaction'][0]-raw[f'{worse}@{s}']['reaction'][0])for s in SEEDS}
            consistent=all(math.isfinite(x)and x>0 for x in seed_deltas.values())
            comparisons.append(dict(better=better,worse=worse,mean_bounds=bounds,
                reaction_delta_by_seed={k:v if math.isfinite(v)else None for k,v in seed_deltas.items()},
                reaction_improves_in_both_seeds=consistent,passes_original_mean_quality_guards=clear,
                robust_preference=clear and consistent))
    point=lambda value:{k:float(v[0])if np.isfinite(v[0])else None for k,v in value.items()}
    return dict(seeds=list(SEEDS),samples_per_seed=len(membership),independent_source_groups=g,
        population_is_not_128_independent_samples=True,common_coverage=info['common_coverage'],issues=info['issues'],
        mean_metrics={n:point(v)for n,v in averaged.items()},
        per_seed_metrics={n:{str(s):point(raw[f'{n}@{s}'])for s in SEEDS}for n in names},comparisons=comparisons,
        bootstrap_draws=policy['bootstrap_draws'],bootstrap_unit='source group; retain both draws together',
        uncertainty_scope='Conditional on these two generation seeds; does not estimate training-seed variance',
        automatic_promotion=False)


def generate_report():
    from p2_loss_search28.config import CANDIDATES
    from stage_driver import verify_scoring
    from dual_seed_queue import work_for,validate_frozen_plan
    plan=validate_frozen_plan();membership=read(WORK/'tuning64.json')
    candidates={};native={};evidence={}
    for name in CANDIDATES:
        candidates[name]={};native[name]={}
        for seed in SEEDS:
            work=work_for(name,seed);receipt=verify_scoring(work);p=read(work/'protocol.json');arm=p['arms'][0]
            if p['seed']!=seed or p['total_samples']!=64 or arm['step']!=500:
                raise ValueError('Wrong checkpoint, seed or population')
            if p['checkpoint_identity']['experiment']['loss_search']['candidate']!=name:
                raise ValueError('Wrong candidate')
            items=read(work/'shared_items.json')
            if {x['id']for x in items}!={x['id']for x in membership}:raise ValueError('Population differs')
            for item in items:
                meta=read(work/'generated'/arm['name']/item['key']/'metadata.json')
                if meta['seed']!=seed or meta['checkpoint_sha256']!=arm['original_sha256']:
                    raise ValueError('Generated media seed/checkpoint differs')
            folder=work/'metrics'/arm['name']/'seamless'
            candidates[name][seed]=dict(events=rows(work/'event_pairs_fixed_reference.jsonl'),
                                       video=rows(folder/'pairs.jsonl'),audio=rows(folder/'audio_pairs.jsonl'))
            native[name][str(seed)]=read(work/'RESULTS.json')[arm['name']+'/seamless']
            evidence[str(work/'SCORING_COMPLETE.json')]=sha(work/'SCORING_COMPLETE.json')
        hashes={read(work_for(name,s)/'export/RECEIPT.json')['original_sha256']for s in SEEDS}
        if len(hashes)!=1:raise ValueError('Seeds used different checkpoints')
    report=compare(candidates,membership,read(WORK/'PROMOTION_POLICY.json'))
    report.update(evidence=evidence,plan_sha256=sha(WORK/'DUAL_SEED_PLAN.json'),
        native_metrics_by_seed=native,native_metric_means={n:{k:strict_mean([native[n][str(s)]['metrics'].get(k)for s in SEEDS])
            for k in set.union(*(set(native[n][str(s)]['metrics'])for s in SEEDS))}for n in native},
        distribution_metric_policy='Compute full64 FID/FVD separately for each seed, then average the two scores; never pool128 features',
        status='comparison_ready_for_review',best_model_selected=False)
    atomic_json(WORK/'DUAL_SEED_COMPARISON.json',report)
    return report


if __name__=='__main__':generate_report()
