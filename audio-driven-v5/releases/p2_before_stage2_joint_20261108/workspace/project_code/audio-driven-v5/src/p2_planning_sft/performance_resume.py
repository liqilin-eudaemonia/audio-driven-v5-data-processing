"""Strict, recorded input-performance migration with unchanged learning semantics."""
import copy
from pathlib import Path
from p2_listen.contracts import read,sha,digest,atomic_json


def validate_performance_resume(config,identity,root):
    policy=config['performance_resume'];checkpoint=Path(policy['checkpoint']).resolve()
    old=read(checkpoint.parent/'run.json')
    if sha(checkpoint.parent/'run.json')!=policy['parent_run_sha256']:raise ValueError('Parent run changed')
    if root.resolve()==checkpoint.parent:raise ValueError('Migration requires a separate output')
    if config.get('extend_from'):raise ValueError('Cannot combine continuation policies')
    for key in ('train_manifest','dev_manifest','train_ids','train_count','dev_count','full_dev_count','reference_data','system','user','smoke'):
        if old.get(key)!=identity.get(key):raise ValueError('Input migration changes training identity: '+key)
    repartition=old['world_size']!=identity['world_size']
    if repartition:
        if old['world_size']!=32 or identity['world_size']!=24 or config['training'].get('global_batch_size')!=256:
            raise ValueError('Unsupported world-size migration')
        if policy.get('rank_rng_sources')!=list(range(8))+list(range(16,32)):
            raise ValueError('Wrong physical-node RNG mapping')
        report_path=policy['batch_partition_report']
        batch_report=read(report_path)
        if sha(report_path)!=policy['batch_partition_report_sha256'] or not batch_report['success']:
            raise ValueError('Missing batch partition verification')
        if batch_report['code_sha256']!=identity['code']['global_batch.py']:
            raise ValueError('Batch partition code changed after verification')
    for name in ('data.py','model.py','continuation.py','__init__.py'):
        if old['code'].get(name)!=identity['code'].get(name):raise ValueError('Training semantics changed: '+name)
    expected=copy.deepcopy(old['config']);expected['output']=config['output']
    expected['data']['input_pipeline']=config['data']['input_pipeline']
    expected['training']['prefetch_factor']=config['training']['prefetch_factor']
    if repartition:expected['training']['global_batch_size']=256
    expected['performance_resume']=policy
    if config!=expected:raise ValueError('Only reader, prefetch and output may change')
    if config['training']['prefetch_factor']!=4:raise ValueError('Unvalidated prefetch')
    report=read(policy['equivalence_report'])
    if sha(policy['equivalence_report'])!=policy['equivalence_report_sha256'] or not report['success']:
        raise ValueError('Input equivalence report changed/failed')
    if report['pipeline']!=config['data']['input_pipeline']:raise ValueError('Untested input pipeline')
    for name,h in report['code'].items():
        if identity['code'].get(name)!=h:raise ValueError('Input code changed after verification: '+name)
    receipt=read(policy['checkpoint_receipt'])
    if receipt['checkpoint']!=str(checkpoint) or receipt['parent_run_sha256']!=policy['parent_run_sha256']:
        raise ValueError('Wrong checkpoint receipt')
    for name,record in receipt['files'].items():
        st=(checkpoint/name).stat()
        if (st.st_size,st.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):raise ValueError('Checkpoint changed: '+name)
    required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(old['world_size'])]
    if any(n not in receipt['files'] for n in required):raise ValueError('Incomplete migration checkpoint')
    state=read(checkpoint/'trainer_state.json')
    if state['global_step']!=policy['step'] or state['max_steps']!=2697:raise ValueError('Unexpected training boundary')
    load_checkpoint=checkpoint
    if repartition:
        load_checkpoint=Path(policy['load_checkpoint']).resolve()
        import os
        for name in required:
            if name.startswith('rng_state_'):continue
            if not os.path.samefile(load_checkpoint/name,checkpoint/name):raise ValueError('Migration state is not the original checkpoint')
        for rank,source_rank in enumerate(policy['rank_rng_sources']):
            if not os.path.samefile(load_checkpoint/f'rng_state_{rank}.pth',checkpoint/f'rng_state_{source_rank}.pth'):
                raise ValueError('RNG alias mismatch')
    return dict(checkpoint=str(load_checkpoint),parent_checkpoint=str(checkpoint),parent_step=state['global_step'],parent_identity=digest(old),
                receipt_sha256=sha(policy['checkpoint_receipt']),input_equivalence_sha256=sha(policy['equivalence_report']))


def audit_loaded_optimizer(trainer,checkpoint):
    steps={int(v['step'].item()) for v in trainer.optimizer.state.values() if 'step' in v}
    target=trainer.performance_migration['parent_step']
    if steps!={target} or trainer.lr_scheduler.last_epoch!=target:raise ValueError('Optimizer/scheduler progress was not restored')
    if trainer.is_world_process_zero():
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        import torch
        saved=load_file(str(Path(checkpoint)/'adapter_model.safetensors'))
        loaded=get_peft_model_state_dict(trainer.accelerator.unwrap_model(trainer.model))
        if saved.keys()!=loaded.keys() or any(not torch.equal(saved[k],loaded[k].detach().cpu()) for k in saved):
            raise ValueError('Loaded adapter differs from checkpoint')
        atomic_json(Path(trainer.args.output_dir)/'performance_load.json',dict(checkpoint=str(checkpoint),
            optimizer_steps=sorted(steps),optimizer_state_entries=len(trainer.optimizer.state),
            scheduler_last_epoch=trainer.lr_scheduler.last_epoch,learning_rates=[g['lr'] for g in trainer.optimizer.param_groups],
            parent_step=target,optimizer_restored=True,adapter_tensors_equal=True))


def audit_loaded_rng(trainer,checkpoint):
    import random
    import numpy as np
    import torch
    rank=trainer.args.process_index
    saved=torch.load(Path(checkpoint)/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
    numpy_state=np.random.get_state();expected=saved['numpy']
    ok=(random.getstate()==saved['python'] and numpy_state[0]==expected[0]
        and np.array_equal(numpy_state[1],expected[1]) and numpy_state[2:]==expected[2:]
        and torch.equal(torch.random.get_rng_state(),saved['cpu'])
        and all(torch.equal(a,b) for a,b in zip(torch.cuda.random.get_rng_state_all(),saved['cuda'])))
    if not ok:raise ValueError('RNG state was not restored exactly')
    atomic_json(Path(trainer.args.output_dir)/'rng_restore'/f'rank-{rank:02d}.json',
                dict(rank=rank,exact=True,checkpoint=str(checkpoint),step=trainer.performance_migration['parent_step']))
