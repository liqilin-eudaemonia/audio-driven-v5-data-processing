"""Prepare an auditable step-2 handoff to immediate duration grouping."""
import copy
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    import torch
    torch.set_num_threads(2)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_listen.contracts import read,sha,atomic_json
    from p2_listen.cache import atomic_torch
    from p2_whisper_av.checkpoint import checked
    from p2_whisper_av.config import source_identity,validate,validate_baseline
    from p2_whisper_av.data import data_identity
    from p2_whisper_av.execution import settings
    from p2_whisper_av.migration import validate_transition,validate_pairs,validate_sources
    from p2_j_duration_execution import make_pairs,Schedule
    from p2_j_duration_benchmark import equal_tensors
    old=ROOT/'experiments/p2_j_qwen_whisper_av_20261104'
    out=ROOT/'experiments/p2_j_qwen_whisper_av_duration_now_20261104'
    out.mkdir(exist_ok=True)
    if (out/'migration.json').exists():raise ValueError('Migration already completed; do not overwrite')
    parent=old/'step_000002.pt';saved=checked(parent);step=saved['step']
    if step!=2 or saved['identity']!=read(old/'run.json'):
        raise ValueError('Expected the verified original step2 checkpoint')
    base=saved['identity']['experiment'];c=copy.deepcopy(base);c['output']=str(out)
    with (Path(c['j_cache'])/'sample_list.csv').open() as stream:
        rows=[dict(requested_duration_s=float(r['duration_s'])) for r in csv.DictReader(stream)]
    spec=c['duration_sampling'];pairs=make_pairs(rows,c['training']['steps'],32,c['training']['seed'],step,'duration',spec['window'])
    manifest=out/'duration.schedule.json'
    atomic_json(manifest,dict(sample_list_sha256=spec['sample_list_sha256'],pairs=pairs),immutable=True)
    spec.update(start_step=step,manifest=str(manifest),sha256=sha(manifest))
    c['whisper_training_execution']['optimizer_state_cpu_offload']=False
    c['whisper_execution_migration']=dict(kind='duration-now-v1',parent_run=str(old),
        parent_checkpoint=str(parent),parent_sha256=sha(parent),prefix_steps=step)
    validate_transition(base,c,step);validate(c);validate_baseline(c,read(c['baseline_config']))
    validate_pairs(Schedule(base).pairs,Schedule(c).pairs,step,32)
    if data_identity(c)!=saved['identity']['data_identity']:raise ValueError('Data identity changed')
    sources=source_identity();changes=validate_sources(saved['identity']['source'],sources,ROOT)
    identity=copy.deepcopy(saved['identity']);identity.update(experiment=c,source=sources,execution=settings(c))
    config=out/'config.json';atomic_json(config,c,immutable=True)
    target=out/parent.name;atomic_torch(target,dict(saved,identity=identity))
    atomic_json(target.with_suffix('.complete.json'),dict(sha256=sha(target),step=step))
    verify=checked(target)
    for key in ('model','optimizer','rank_rng'):
        if not equal_tensors(saved[key],verify[key]):raise ValueError('Migration changed '+key)
    atomic_json(out/'run.json',identity,immutable=True)
    (out/'metrics.jsonl').write_text(''.join(line+'\n' for line in (old/'metrics.jsonl').read_text().splitlines()
        if __import__('json').loads(line)['step']<=step))
    validation=read(old/'GPU_VALIDATION.json');validation.update(inherited_from=str(old/'GPU_VALIDATION.json'),
        migration_longest_sample_recheck=True)
    atomic_json(out/'GPU_VALIDATION.json',validation,immutable=True)
    atomic_json(out/'status.json',dict(step=step,budget=8927,complete=False))
    atomic_json(out/'migration.json',dict(source=str(parent),source_sha256=sha(parent),step=step,
        model_optimizer_rng_bitwise_unchanged=True,consumed_prefix_identical=True,
        remaining_sample_noise_membership_identical=True,data_identity_identical=True,
        source_changes=changes,changes=['duration grouping from step3','GPU-resident Adam moments','first16-step timing'],
        script_sha256=sha(Path(__file__))),immutable=True)
    queue=out/'queue';queue.mkdir(exist_ok=True)
    spec=read(old/'queue/queue.json');spec.update(config=str(config),config_sha256=sha(config),output=str(out))
    atomic_json(queue/'queue.json',spec,immutable=True)
    atomic_json(queue/'state.json',dict(phase='prepared_duration_now',resume_step=step))
    atomic_json(old/'CURRENT_RUN.json',dict(output=str(out),config=str(config),resume_step=step,
        reason='User requested immediate duration grouping; original checkpoint retained'))
    print(str(queue),flush=True)


if __name__=='__main__':main()
