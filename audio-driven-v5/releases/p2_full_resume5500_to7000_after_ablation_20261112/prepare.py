"""CPU-only preparation and explicit31-to28 migration of original5500 state."""
import copy, math, time
from pathlib import Path
import run_runtime as R
from p2_listen.contracts import atomic_json, read, digest, sha

def equal(a, b):
    import torch
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b

def main():
    import torch
    torch.set_num_threads(2)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_ltx_all_lora.schedule import Schedule as Native
    from p2_ltx_all_lora.config import source_identity
    from p2_ltx_all_lora.data import data_identity
    from p2_ltx_all_lora import checkpoint
    from p2_listen.cache import atomic_torch
    parent = read(R.PARENT/'run.json')
    current = source_identity()
    assert all(current.get(k) == v for k, v in parent['source'].items()), 'Parent training source changed'
    base = read(R.PARENT/'config.json')
    native = Native(base)
    plan = dict(start_step=5500, prefix_pairs=5500*31, total_pairs=len(native.pairs),
                original_steps=base['training']['steps'],
                new_end_step=5500+math.ceil((len(native.pairs)-5500*31)/28),
                parent_config=str(R.PARENT/'config.json'), parent_config_sha256=sha(R.PARENT/'config.json'))
    atomic_json(R.W/'PLAN.json', plan, immutable=True)
    c = copy.deepcopy(base)
    c['output'] = str(R.OUT)
    c['training'].update(global_batch=28, save_every=100, steps=plan['new_end_step'])
    c['batch28_continuation'] = plan
    R.validate(c)
    R.OUT.mkdir(exist_ok=True)
    (R.OUT/'rank_logs').mkdir(exist_ok=True)
    atomic_json(R.OUT/'config.json', c, immutable=True)
    assert data_identity(R.normalized(c)) == parent['data_identity']
    schedule = R.Schedule(c)
    selected = [pair for s in range(5501,7001) for pair in schedule.indices(schedule.size,s,28,schedule.seed)]
    assert selected == native.pairs[5500*31:5500*31+1500*28]
    assert all(len(schedule.indices(schedule.size,s,28,schedule.seed,r,28)) == 1
               for s in [5501,5600,6000,7000] for r in range(28))
    assert [s for s in range(5501,7001) if s % c['training']['save_every'] == 0] == list(range(5600,7001,100))
    pre = copy.deepcopy(read(R.PARENT/'preflight.json'))
    pre.update(configuration_sha256=digest(c), source=R.source())
    atomic_json(R.OUT/'preflight.json', pre, immutable=True)
    cluster = copy.deepcopy(read(R.W.parent/'p2_ltx_all_lora_full28_batch28_20261109/cluster.json'))
    cluster.update(master_port=29983, training_script=str(R.W/'worker.py'), training_config=str(R.OUT/'config.json'))
    assert [n['name'] for n in cluster['nodes']] == ['node12','node8','node4','node11']
    assert [n['gpus'] for n in cluster['nodes']] == [list(range(8))]*3+[list(range(4))]
    atomic_json(R.OUT/'cluster.json', cluster, immutable=True)
    source = R.PARENT/'step_005500.pt'
    dest = R.OUT/source.name
    print('Checking original5500 model/Adam/31RNG and checksum', flush=True)
    old = checkpoint.checked(source)
    assert old['step'] == 5500 and old['identity'] == parent and len(old['rank_rng']) == 31
    identity = copy.deepcopy(parent)
    identity.update(experiment=c, source=R.source(), world=28)
    payload = dict(old, identity=identity, rank_rng=[copy.deepcopy(old['rank_rng'][0]) for _ in range(28)])
    if not dest.exists():
        atomic_torch(dest, payload)
        atomic_json(dest.with_suffix('.complete.json'), dict(step=5500,sha256=sha(dest)))
    new = checkpoint.checked(dest)
    assert new['identity'] == identity and new['step'] == 5500 and len(new['rank_rng']) == 28
    assert equal(old['model'], new['model']) and equal(old['optimizer'], new['optimizer'])
    assert all(equal(r, old['rank_rng'][0]) for r in new['rank_rng'])
    atomic_json(R.OUT/'run.json', identity, immutable=True)
    atomic_json(R.OUT/'MIGRATION.json', dict(success=True, source=str(source), destination=str(dest),
                source_sha256=read(source.with_suffix('.complete.json'))['sha256'],
                destination_sha256=read(dest.with_suffix('.complete.json'))['sha256'],
                step=5500, parent_world=31, world=28, model_and_adam_equal=True,
                rng_policy='Parent rank0 RNG cloned to28; source31 retained; per-sample ordinals preserved'))
    atomic_json(R.W/'READY.json', dict(success=True, at=time.time(), source=R.source(),
                config_sha256=sha(R.OUT/'config.json'), cluster_sha256=sha(R.OUT/'cluster.json'),
                source_checkpoint_sha256=read(source.with_suffix('.complete.json'))['sha256'],
                start=5500, target=7000, world=28, batch=28, save_every=100,
                continuous_sample_noise_ordinals_verified=True, model_and_adam_equal=True,
                expected_saves=list(range(5600,7001,100))))
    print('Preparation and full-state migration accepted', flush=True)

if __name__ == '__main__': main()
