"""Strict second continuation: preserve the unconsumed batch31 schedule on24 ranks."""
import copy
from p2_listen.contracts import read,sha
from .resume31 import remaining_plan


def continuation_plan(parent_schedule,resume_step):
    relative=resume_step-parent_schedule.base_step
    if type(resume_step) is not int or relative<1:
        raise ValueError('Resume must follow the parent continuation base')
    plan=remaining_plan(parent_schedule.pairs,relative,31,24,
                        parent_schedule.durations,parent_schedule.seed)
    plan.update(schema='resume24-parent-suffix-v1',resume_step=resume_step,
                parent_base_step=parent_schedule.base_step,parent_consumed_updates=relative)
    return plan


def validate_resume24(c):
    spec=c['resume_execution']
    required={'kind','parent_config','parent_config_sha256','resume_step','manifest',
              'manifest_sha256','world','parent_world'}
    if (set(spec)!=required or spec['kind']!='preserve-parent-suffix-v1'
            or spec['world']!=24 or spec['parent_world']!=31
            or type(spec['resume_step']) is not int or spec['resume_step']<1
            or c['training']['global_batch']!=24):
        raise ValueError('Invalid24-rank continuation')
    if sha(spec['parent_config'])!=spec['parent_config_sha256']:
        raise ValueError('Frozen31-rank parent config changed')
    parent=read(spec['parent_config'])
    from .resume31 import validate_resume
    validate_resume(parent)
    if parent['training']['global_batch']!=31:
        raise ValueError('Require the accepted31-rank parent')
    expected=copy.deepcopy(parent)
    expected.update(output=c['output'],resume_execution=spec,
                    world24_execution=dict(kind='resume24-preserve-remaining-v1',world=24,
                                           reserved_node='node11',reserved_gpus=list(range(8))))
    expected['training'].update(global_batch=24,steps=c['training']['steps'])
    if c!=expected:raise ValueError('24-rank continuation changed unrelated training choices')
    return parent


def select_indices(pairs,step,base,rank,world,batch):
    if world!=24 or batch!=24 or type(rank) is not int or not 0<=rank<24:
        raise ValueError('Expected24 ranks and batch24')
    offset=step-base-1
    if offset<0 or offset>=len(pairs)//24:raise ValueError('Step outside remaining schedule')
    return [tuple(pairs[offset*24+rank])]


class Resume24Schedule:
    def __init__(self,c):
        from .schedule import Schedule,CurrentSchedule
        parent=validate_resume24(c);original=Schedule(parent)
        # The batch31 schedule carries a suffix; its durations come from the
        # original frozen dataset, never from the suffix ordinal positions.
        original.durations=CurrentSchedule(read(parent['resume_execution']['parent_config'])).durations
        spec=c['resume_execution']
        if sha(spec['manifest'])!=spec['manifest_sha256']:raise ValueError('24-rank plan changed')
        plan=read(spec['manifest'])
        if plan!=continuation_plan(original,spec['resume_step']):
            raise ValueError('24-rank schedule lost/replayed sample-noise pairs')
        self.pairs=[tuple(p) for p in plan['pairs']];self.base_step=spec['resume_step']
        self.batch=24;self.size=original.size;self.seed=original.seed
        if c['training']['steps']!=self.base_step+len(self.pairs)//24:
            raise ValueError('24-rank budget does not cover the suffix')

    def indices(self,size,step,batch,seed,rank=0,world=1):
        if (size,seed)!=(self.size,self.seed):raise ValueError('Dataset/seed differs')
        return select_indices(self.pairs,step,self.base_step,rank,world,batch)
