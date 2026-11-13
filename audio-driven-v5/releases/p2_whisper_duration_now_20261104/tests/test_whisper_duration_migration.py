import copy
from pathlib import Path

import pytest

from p2_whisper_av.config import load
from p2_whisper_av.migration import validate_transition,validate_pairs,validate_sources


def migration():
    old=load();old['whisper_training_execution']=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=32,
        optimizer_state_cpu_offload=True,longest_sample_check=True)
    new=copy.deepcopy(old);new['output']+='_duration'
    new['duration_sampling'].update(start_step=2,manifest='new.json',sha256='b'*64)
    new['whisper_training_execution']['optimizer_state_cpu_offload']=False
    new['whisper_execution_migration']=dict(kind='duration-now-v1',parent_run='parent',
        parent_checkpoint='parent/step_000002.pt',parent_sha256='a'*64,prefix_steps=2)
    return old,new


def test_only_authorized_execution_changes_accepted():
    old,new=migration();validate_transition(old,new,2)
    for key,value in [('steps',8000),('global_batch',16),('connector_lr',1e-6)]:
        bad=copy.deepcopy(new);bad['training'][key]=value
        with pytest.raises(ValueError):validate_transition(old,bad,2)
    for key,value in [('window',1024),('sample_list_sha256','changed'),('start_step',3)]:
        bad=copy.deepcopy(new);bad['duration_sampling'][key]=value
        with pytest.raises(ValueError):validate_transition(old,bad,2)


def test_consumed_prefix_and_all_occurrences_are_preserved():
    old=[(i,i+100) for i in range(12)];new=old[:4]+list(reversed(old[4:]))
    validate_pairs(old,new,2,2)
    for bad in [list(reversed(new)),new[:-1]+[new[-2]],new[:-1]+[(99,99)]]:
        with pytest.raises(ValueError):validate_pairs(old,bad,2,2)


def test_source_changes_are_explicitly_bounded():
    root=Path('/project');path=str(root/'src/p2_whisper_av/training.py')
    assert validate_sources({path:'old'},{path:'new'},root)[path]['before']=='old'
    with pytest.raises(ValueError):validate_sources({'other.py':'old'},{'other.py':'new'},root)
