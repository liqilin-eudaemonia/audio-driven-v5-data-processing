import copy
import json
from pathlib import Path
import pytest
from p2_listen.contracts import sha
from p2_planning_sft.performance_resume import validate_performance_resume


@pytest.fixture
def migration(tmp_path):
    parent=tmp_path/'old';cp=parent/'checkpoint-500';cp.mkdir(parents=True)
    old=dict(config=dict(output=str(parent),data=dict(mode='reference_jsonl'),training=dict(epochs=1,micro_batch=8,learning_rate=1e-4)),
             train_ids='ids',train_count=690337,world_size=32,system='s',user='u',smoke=False,
             code={n:'original' for n in ('data.py','model.py','continuation.py','__init__.py')})
    (parent/'run.json').write_text(json.dumps(old))
    state=dict(global_step=500,max_steps=2697);(cp/'trainer_state.json').write_text(json.dumps(state))
    for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt']+[f'rng_state_{i}.pth' for i in range(32)]:(cp/name).write_bytes(b'checkpoint')
    files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in cp.iterdir()}
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(dict(checkpoint=str(cp),parent_run_sha256=sha(parent/'run.json'),files=files)))
    pipeline=dict(version='tested');report=tmp_path/'report.json';report.write_text(json.dumps(dict(success=True,pipeline=pipeline,code={'fast_media.py':'new'})))
    c=copy.deepcopy(old['config']);c['output']=str(tmp_path/'new');c['data']['input_pipeline']=pipeline;c['training']['prefetch_factor']=4
    c['performance_resume']=dict(checkpoint=str(cp),parent_run_sha256=sha(parent/'run.json'),equivalence_report=str(report),equivalence_report_sha256=sha(report),checkpoint_receipt=str(receipt),step=500)
    identity=copy.deepcopy(old);identity['config']=c;identity['code']['fast_media.py']='new'
    return c,identity,tmp_path/'new',cp,receipt


def test_migration_preserves_budget_and_checkpoint(migration):
    c,i,out,cp,_=migration;r=validate_performance_resume(c,i,out)
    assert r['parent_step']==500 and r['checkpoint']==str(cp)


@pytest.mark.parametrize('key,value',[('micro_batch',16),('learning_rate',2e-4),('epochs',2)])
def test_training_parameter_change_rejected(migration,key,value):
    c,i,out,_,_=migration;c['training'][key]=value
    with pytest.raises(ValueError,match='Only reader'):validate_performance_resume(c,i,out)


def test_missing_rank_rng_rejected(migration):
    c,i,out,cp,receipt=migration
    record=json.loads(receipt.read_text());del record['files']['rng_state_31.pth'];receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Incomplete'):validate_performance_resume(c,i,out)


def test_changed_model_code_rejected(migration):
    c,i,out,_,_=migration;i['code']['model.py']='changed'
    with pytest.raises(ValueError,match='semantics'):validate_performance_resume(c,i,out)


def test_changed_checkpoint_rejected(migration):
    c,i,out,cp,_=migration;(cp/'optimizer.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Checkpoint changed'):validate_performance_resume(c,i,out)


def test_24_rank_migration_requires_same_global_batches_and_physical_rng_mapping(migration,tmp_path):
    import os
    c,i,out,cp,_=migration
    c['training']['global_batch_size']=256;i['world_size']=24;i['code']['global_batch.py']='verified'
    report=tmp_path/'batches.json';report.write_text(json.dumps(dict(success=True,code_sha256='verified')))
    mapping=list(range(8))+list(range(16,32));alias=tmp_path/'alias';alias.mkdir()
    for p in cp.iterdir():
        if not p.name.startswith('rng_state_'):os.link(p,alias/p.name)
    for rank,source in enumerate(mapping):os.link(cp/f'rng_state_{source}.pth',alias/f'rng_state_{rank}.pth')
    c['performance_resume'].update(rank_rng_sources=mapping,load_checkpoint=str(alias),batch_partition_report=str(report),batch_partition_report_sha256=sha(report))
    assert validate_performance_resume(c,i,out)['checkpoint']==str(alias)
    c['performance_resume']['rank_rng_sources']=list(range(24))
    with pytest.raises(ValueError,match='RNG mapping'):validate_performance_resume(c,i,out)
