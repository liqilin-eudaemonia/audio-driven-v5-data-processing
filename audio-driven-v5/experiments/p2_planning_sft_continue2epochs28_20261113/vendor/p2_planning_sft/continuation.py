"""Extend a completed run by one epoch while preserving optimizer history."""
import copy
import math
from pathlib import Path

from p2_listen.contracts import read, digest


def validate_extension(config, identity, root):
    policy = config['extension_policy']
    checkpoint = Path(config['extend_from']).absolute()
    source_checkpoint = Path(policy['parent_checkpoint']).resolve()
    parent = source_checkpoint.parent
    old = read(parent/'run.json')
    complete = read(parent/'COMPLETE.json')
    state = read(checkpoint/'trainer_state.json')
    if root.resolve() == parent or not complete['success'] or complete['smoke'] != identity['smoke']:
        raise ValueError('Extension requires a completed matching run in a separate directory')
    if state['global_step'] != complete['global_step'] or state['global_step'] != state['max_steps']:
        raise ValueError('Extend from the final completed checkpoint, not an intermediate checkpoint')
    for key in ('train_manifest', 'dev_manifest', 'train_ids', 'train_count', 'dev_count', 'system', 'user'):
        if old[key] != identity[key]:
            raise ValueError('Extension changes data/prompt/world: '+key)
    for key in old['code']:
        if key in ('continuation.py', 'train.py'):
            continue
        if old['code'][key] != identity['code'][key]:
            raise ValueError('Extension changes input/target/loss implementation')
    desired = copy.deepcopy(old['config'])
    desired.pop('performance_resume')
    desired['output'] = config['output']
    desired['extend_from'] = config['extend_from']
    desired['extension_policy'] = policy
    desired['initialization'] = 'extend_completed_sft_preserve_optimizer'
    desired['training']['epochs'] += policy['additional_epochs']
    desired['training']['micro_batch'] = 10
    desired['data']['seal'] = config['data']['seal']
    if config != desired:
        raise ValueError('Unexpected extension hyperparameter/input change')
    assert policy['additional_epochs'] == 2
    assert old['world_size'] == policy['parent_world_size'] == 32
    assert identity['world_size'] == policy['world_size'] == 28
    assert identity['train_count'] == 704368
    old_seal = read(old['config']['data']['seal'])
    new_seal = read(config['data']['seal'])
    old_vendor = str(parent.parent/'vendor')
    new_vendor = str(root.parent/'vendor')
    expected_seal = copy.deepcopy(old_seal)
    expected_seal['reader_code'] = {k.replace(old_vendor, new_vendor):v for k,v in old_seal['reader_code'].items()}
    assert new_seal == expected_seal, 'Data or reader semantics changed'
    reference = copy.deepcopy(identity['reference_data'])
    for name, entry in reference.items():
        entry['seal_sha256'] = old['reference_data'][name]['seal_sha256']
        entry['reader_code'] = {k.replace(new_vendor, old_vendor):v for k,v in entry['reader_code'].items()}
    assert reference == old['reference_data'], 'Reference data changed'
    for name in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt'):
        if not (checkpoint/name).is_file():
            raise ValueError('Incomplete continuation checkpoint: '+name)
    # Keep the original warmup count; the new budget only extends cosine decay.
    warmup = old.get('extension', {}).get('warmup_steps')
    if warmup is None:
        warmup = math.ceil(state['max_steps'] * old['config']['training']['warmup_ratio'])
    return dict(checkpoint=str(checkpoint), source_checkpoint=str(source_checkpoint), parent_identity=digest(old),
                parent_step=state['global_step'], parent_epochs=old['config']['training']['epochs'],
                warmup_steps=warmup, schedule='extend_cosine_horizon_preserve_optimizer')


def align_loaded_learning_rate(optimizer, scheduler):
    """Refresh the zero terminal LR using the new horizon, without taking a step."""
    rates = [base * fn(scheduler.last_epoch)
             for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    if not all(rate > 0 for rate in rates):
        raise ValueError('Extended schedule must have a nonzero starting learning rate')
    for group, rate in zip(optimizer.param_groups, rates):
        group['lr'] = rate
    scheduler._last_lr = rates
    return rates
