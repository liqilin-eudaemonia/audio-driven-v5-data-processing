"""Audited one-to-32-rank migration of the current upright-label LoRA run."""
import copy
import os
from pathlib import Path
from p2_listen.contracts import read, sha, digest, atomic_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_performance_resume(config, identity, root):
    policy = config['performance_resume']
    parent_checkpoint = Path(policy['parent_checkpoint']).resolve()
    checkpoint = Path(policy['load_checkpoint']).resolve()
    parent = parent_checkpoint.parent
    old = read(parent/'run.json')
    require(sha(parent/'run.json') == policy['parent_run_sha256'], 'Parent identity changed')
    require(old['world_size'] == 1 and identity['world_size'] == 32, 'Expected one-to-32-rank migration')
    require(root.resolve() != parent, 'Preserve parent output')
    require(identity['code'] == policy['validated_code'], 'Unvalidated migration code')
    for name, value in old['code'].items():
        if name not in ('train.py', 'global_batch.py', 'performance_resume.py'):
            require(identity['code'].get(name) == value, 'Model/data semantics changed: '+name)
    for name in ('train_manifest', 'dev_manifest', 'train_ids', 'train_count', 'dev_count',
                 'full_dev_count', 'system', 'user', 'smoke'):
        require(identity.get(name) == old.get(name), 'Training identity changed: '+name)
    require(identity['train_count'] == 704368 and not identity['smoke'], 'Wrong training selection')
    old_seal = read(old['config']['data']['seal'])
    new_seal = read(config['data']['seal'])
    expected_seal = copy.deepcopy(old_seal)
    expected_seal['reader_code'] = {
        k.replace(policy['parent_vendor'], policy['new_vendor']):v
        for k,v in old_seal['reader_code'].items()}
    require(new_seal == expected_seal, 'Derived seal changed data or media readers')
    require(sha(config['data']['seal']) == policy['derived_seal_sha256'], 'Derived seal changed')
    ref = copy.deepcopy(identity['reference_data'])
    for name, entry in ref.items():
        entry['seal_sha256'] = old['reference_data'][name]['seal_sha256']
        entry['reader_code'] = {
            k.replace(policy['new_vendor'], policy['parent_vendor']):v
            for k,v in entry['reader_code'].items()}
    require(ref == old['reference_data'], 'Migration changes reference inputs')
    expected = copy.deepcopy(old['config'])
    expected['output'] = config['output']
    expected['initialization'] = 'resume_upright_checkpoint_with_optimizer_and_scheduler'
    expected['training'].update(gradient_accumulation=1, global_batch_size=256, sampler_group_batch=256)
    expected['data']['seal'] = config['data']['seal']
    expected['performance_resume'] = policy
    require(config == expected, 'Unsupported hyperparameter/input change')
    require(old['config']['training']['micro_batch'] * old['config']['training']['gradient_accumulation'] == 256,
            'Wrong parent global batch')
    state = read(parent_checkpoint/'trainer_state.json')
    require(state['global_step'] == policy['step'] == 1500 and state['max_steps'] == 2752,
            'Wrong parent training boundary')
    receipt = read(policy['checkpoint_receipt'])
    require(sha(policy['checkpoint_receipt']) == policy['checkpoint_receipt_sha256'], 'Checkpoint receipt changed')
    for name, record in receipt['files'].items():
        p = parent_checkpoint/name
        st = p.stat()
        require((st.st_size, st.st_mtime_ns) == (record['bytes'], record['mtime_ns']), 'Parent checkpoint changed: '+name)
        require(os.path.samefile(p, checkpoint/name), 'Checkpoint state was not preserved: '+name)
    rank = int(os.environ['RANK'])
    rng = checkpoint/f'rng_state_{rank}.pth'
    require(sha(rng) == policy['rank_rng_sha256'][str(rank)], 'Rank RNG mapping changed')
    report = read(policy['batch_partition_report'])
    require(sha(policy['batch_partition_report']) == policy['batch_partition_report_sha256'] and report['passed'],
            'Missing partition verification')
    require(report['code_sha256'] == identity['code']['global_batch.py'], 'Partition code changed')
    require(sha(policy['input_equivalence']) == policy['input_equivalence_sha256'], 'Parent input acceptance changed')
    require(read(policy['input_equivalence'])['passed'], 'Parent input acceptance failed')
    return dict(checkpoint=str(checkpoint), parent_checkpoint=str(parent_checkpoint), parent_step=1500,
                parent_identity=digest(old), world_size_change=[1,32], global_batch=256,
                parent_cuda_rng_expanded_to_local_devices=8, tail_padding=False)


def audit_loaded_optimizer(trainer, checkpoint):
    steps = {int(v['step'].item()) for v in trainer.optimizer.state.values() if 'step' in v}
    target = trainer.performance_migration['parent_step']
    require(steps == {target} and trainer.lr_scheduler.last_epoch == target,
            'Optimizer/scheduler was not restored')
    record = dict(rank=trainer.args.process_index, checkpoint=str(checkpoint),
        optimizer_steps=sorted(steps), optimizer_state_entries=len(trainer.optimizer.state),
        scheduler_last_epoch=trainer.lr_scheduler.last_epoch,
        learning_rates=[g['lr'] for g in trainer.optimizer.param_groups], parent_step=target,
        optimizer_restored=True)
    require(record['optimizer_state_entries'] == 504, 'Incomplete Adam state')
    atomic_json(Path(trainer.args.output_dir)/'optimizer_restore'/f'rank-{trainer.args.process_index:02d}.json', record)
    if trainer.is_world_process_zero():
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        saved = load_file(str(Path(checkpoint)/'adapter_model.safetensors'))
        loaded = get_peft_model_state_dict(trainer.accelerator.unwrap_model(trainer.model))
        require(saved.keys() == loaded.keys() and all(torch.equal(saved[k], loaded[k].detach().cpu()) for k in saved),
                'Restored adapter differs from parent checkpoint')
        atomic_json(Path(trainer.args.output_dir)/'performance_load.json', dict(record, adapter_tensors_equal=True))


def audit_loaded_rng(trainer, checkpoint):
    import random
    import numpy as np
    import torch
    rank = trainer.args.process_index
    saved = torch.load(Path(checkpoint)/f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
    actual = np.random.get_state()
    expected = saved['numpy']
    ok = (random.getstate() == saved['python'] and actual[0] == expected[0]
          and np.array_equal(actual[1], expected[1]) and actual[2:] == expected[2:]
          and torch.equal(torch.random.get_rng_state(), saved['cpu'])
          and len(torch.cuda.random.get_rng_state_all()) == len(saved['cuda'])
          and all(torch.equal(a,b) for a,b in zip(torch.cuda.random.get_rng_state_all(), saved['cuda'])))
    require(ok, 'Expanded parent RNG was not restored exactly')
    atomic_json(Path(trainer.args.output_dir)/'rng_restore'/f'rank-{rank:02d}.json',
                dict(rank=rank, exact_to_migration_payload=True, checkpoint=str(checkpoint), step=1500))
