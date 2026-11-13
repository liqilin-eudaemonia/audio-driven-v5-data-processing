"""Operational32->31 continuation; preserve sample/noise pairs and duration grouping."""
import copy,random
from pathlib import Path
from p2_listen.contracts import read,sha


def validate_checkpoint_schedule(c):
    policy=c.get('checkpoint_schedule')
    if policy is None:return
    if (set(policy)!={'threshold_step','before_every','after_every'}
            or any(type(v) is not int or v<1 for v in policy.values())
            or policy['threshold_step']%policy['after_every']
            or c['training']['save_every']!=policy['after_every']):
        raise ValueError('Invalid explicit checkpoint cadence')


def checkpoint_due(c,step):
    policy=c.get('checkpoint_schedule')
    interval=c['training']['save_every'] if policy is None else policy[
        'before_every' if step<policy['threshold_step'] else 'after_every']
    return step%interval==0


def remaining_plan(parent_pairs,resume_step,parent_batch,batch,durations=None,seed=0):
    if any(type(v) is not int or v<1 for v in (resume_step,parent_batch,batch)):
        raise ValueError('Invalid continuation geometry')
    pairs=[tuple(p) for p in parent_pairs]
    start=resume_step*parent_batch
    if start>=len(pairs):raise ValueError('Continuation has no unconsumed samples')
    remaining=pairs[start:];padding=(-len(remaining))%batch
    next_ordinal=max(p[1] for p in pairs)+1
    extra=[(remaining[i%len(remaining)][0],next_ordinal+i) for i in range(padding)]
    selected=remaining+extra
    plan=dict(schema='resume31-suffix-v1',resume_step=resume_step,parent_batch=parent_batch,
              batch=batch,original_remaining=len(remaining),padding=padding)
    if durations is not None:
        # Preserve every unconsumed (sample, noise ordinal) pair; restore narrow
        # duration batches after changing geometry from32 to31.
        window=batch*16;ordered=[]
        for offset in range(0,len(selected),window):
            block=sorted(selected[offset:offset+window],key=lambda p:durations[p[0]])
            groups=[block[i:i+batch] for i in range(0,len(block),batch)]
            random.Random(seed+104729+start+offset).shuffle(groups)
            ordered.extend(p for group in groups for p in group)
        selected=ordered
        plan.update(ordering='duration-rebatch-v1',window=window,seed=seed)
    plan['pairs']=[list(p) for p in selected]
    return plan


def validate_resume(c):
    spec=c.get('resume_execution')
    if spec is None:return None
    required={'kind','parent_config','parent_config_sha256','resume_step','manifest',
              'manifest_sha256','world','parent_world'}
    if (set(spec)!=required or spec['kind']!='preserve-parent-suffix-v1'
            or spec['world']!=31 or spec['parent_world']!=32
            or type(spec['resume_step']) is not int or spec['resume_step']<1
            or c['training']['global_batch']!=31):
        raise ValueError('Invalid explicit world31 continuation')
    if sha(spec['parent_config'])!=spec['parent_config_sha256']:
        raise ValueError('Continuation parent configuration changed')
    parent=read(spec['parent_config'])
    if parent['training']['global_batch']!=32:
        raise ValueError('Continuation requires the frozen batch32 parent')
    expected=copy.deepcopy(parent)
    expected.update(output=c['output'],resume_execution=spec,
                    checkpoint_schedule=c['checkpoint_schedule'],
                    world31_execution=c['world31_execution'])
    expected['training'].update(global_batch=31,steps=c['training']['steps'],
                                save_every=c['training']['save_every'])
    meta=c['world31_execution']
    if (c!=expected or meta!=dict(kind='resume31-preserve-remaining-v1',world=31,
                                 reserved_node='node11',reserved_gpu=7)):
        raise ValueError('Continuation changed an unrelated training choice')
    return parent


def select_indices(pairs,step,resume_step,batch,rank=0,world=31):
    if world!=31 or batch!=31 or type(rank) is not int or not 0<=rank<world:
        raise ValueError('World31 continuation geometry differs')
    relative=step-resume_step-1
    if relative<0 or relative>=len(pairs)//batch:
        raise ValueError('Continuation step outside the unconsumed suffix')
    return [tuple(pairs[relative*batch+rank])]


class CurrentResumeSchedule:
    def __init__(self,c):
        from .schedule import CurrentSchedule
        parent=validate_resume(c)
        if parent is None:raise ValueError('Explicit continuation contract required')
        original=CurrentSchedule(parent);spec=c['resume_execution']
        if sha(spec['manifest'])!=spec['manifest_sha256']:
            raise ValueError('Continuation plan changed')
        plan=read(spec['manifest'])
        expected=remaining_plan(original.pairs,spec['resume_step'],32,31,
                                original.durations,original.seed)
        if plan!=expected:raise ValueError('Continuation lost or reordered sample/noise pairs')
        self.pairs=[tuple(p) for p in plan['pairs']]
        self.base_step=spec['resume_step'];self.batch=31
        self.size=original.size;self.seed=original.seed
        if c['training']['steps']!=self.base_step+len(self.pairs)//31:
            raise ValueError('Continuation budget does not cover the exact suffix')

    def indices(self,size,step,batch,seed,rank=0,world=1):
        if (size,batch,seed)!=(self.size,self.batch,self.seed):
            raise ValueError('Continuation dataset/batch/seed changed')
        return select_indices(self.pairs,step,self.base_step,batch,rank,world)
