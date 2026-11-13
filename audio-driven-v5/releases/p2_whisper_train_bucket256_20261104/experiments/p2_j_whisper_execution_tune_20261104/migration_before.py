"""Explicit execution-only handoff; tensors and the consumed sample prefix stay fixed."""
import copy
import re
from collections import Counter


def validate_transition(parent, target, step):
    meta=target.get('whisper_execution_migration',{})
    if (set(meta)!={'kind','parent_run','parent_checkpoint','parent_sha256','prefix_steps'}
            or meta['kind']!='duration-now-v1' or meta['prefix_steps']!=step
            or type(step) is not int or not 0<step<parent['duration_sampling']['start_step']
            or re.fullmatch('[0-9a-f]{64}',meta['parent_sha256']) is None):
        raise ValueError('Invalid explicit Whisper execution migration')
    before=parent['duration_sampling'];after=target['duration_sampling']
    if (set(before)!=set(after) or after['start_step']!=step
            or any(after[k]!=before[k] for k in before if k not in ('start_step','manifest','sha256'))):
        raise ValueError('Migration changed duration policy, window or sample membership')
    if after['policy']!='duration' or target['output']==parent['output']:
        raise ValueError('Migration requires duration scheduling and a new output')
    execution=copy.deepcopy(target['whisper_training_execution'])
    execution['optimizer_state_cpu_offload']=parent['whisper_training_execution']['optimizer_state_cpu_offload']
    if execution!=parent['whisper_training_execution']:
        raise ValueError('Migration changed unsupported execution settings')
    normalized=copy.deepcopy(target);normalized.pop('whisper_execution_migration')
    for key in ('output','duration_sampling','whisper_training_execution'):
        normalized[key]=copy.deepcopy(parent[key])
    if normalized!=parent:
        raise ValueError('Migration changed data, model, optimizer or objective')


def validate_pairs(before, after, prefix_steps, batch):
    old=[tuple(p) for p in before];new=[tuple(p) for p in after]
    if old[:prefix_steps*batch]!=new[:prefix_steps*batch] or Counter(old)!=Counter(new):
        raise ValueError('Migration changed consumed prefix or remaining sample/noise membership')


def validate_sources(before, after, root):
    allowed={str(root/'src/p2_whisper_av'/name) for name in ('config.py','training.py','migration.py')}
    changed={key for key in set(before)|set(after) if before.get(key)!=after.get(key)}
    if not changed or not changed<=allowed:
        raise ValueError('Unexpected training source changes during execution migration')
    return {key:dict(before=before.get(key),after=after.get(key)) for key in sorted(changed)}
