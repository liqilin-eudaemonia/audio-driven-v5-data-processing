"""Versioned 31->30 continuation; native model/loss/optimizer code stays immutable."""
import copy,csv,random,math
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha,digest


def make_remaining(parent,prefix):
    from p2_j_duration_execution import Schedule as Native
    path=Path(parent['j_cache'])/'sample_list.csv'
    with path.open() as f:durations=[float(r['duration_s']) for r in csv.DictReader(f)]
    full=Native(read(parent['ablation_reference']['config'])).pairs
    original=full[:248000];consumed=prefix*31;remainder=original[consumed:]
    if len({p[1] for p in original})!=len(original):raise ValueError('Original prefix is not unique')
    pad=(-len(remainder))%30;extras=[p for p in full if p[1]>=len(durations)][:pad]
    if len(extras)!=pad or any(p[1]<len(durations) for p in extras):raise ValueError('Missing deterministic tail padding')
    source=remainder+extras;aligned=[]
    for offset in range(0,len(source),480):
        block=sorted(source[offset:offset+480],key=lambda p:durations[p[0]])
        groups=[block[i:i+30] for i in range(0,len(block),30)]
        random.Random(parent['training']['seed']+104729+consumed+offset).shuffle(groups)
        aligned.extend(p for group in groups for p in group)
    if sorted(aligned)!=sorted(source):raise ValueError('Remaining sample/noise pairs changed')
    return aligned,dict(consumed=consumed,remaining=len(remainder),padding=pad,
        remaining_sha256=digest(remainder),consumed_sha256=digest(original[:consumed]),window=480,
        total_original_exposures=248000,steps=prefix+len(aligned)//30)


def normalized(c):
    meta=c['continuation30'];parent=read(meta['parent_config'])
    if sha(meta['parent_config'])!=meta['parent_config_sha256']:raise ValueError('Parent changed')
    prefix=4000 if c['loss']=='av_event' else 0
    if meta['prefix_step']!=prefix or meta['rng_parent_ranks']!=list(range(24,31))+[7]+list(range(8,24))+list(range(6)):
        raise ValueError('Unexpected prefix/RNG mapping')
    expected=copy.deepcopy(parent);expected.update(output=c['output'],continuation30=meta)
    expected['training'].update(global_batch=30,steps=prefix+math.ceil((248000-prefix*31)/30))
    expected['duration_sampling']=c['duration_sampling']
    ds=c['duration_sampling']
    allowed_ds=copy.deepcopy(parent['duration_sampling']);allowed_ds.update({k:ds[k] for k in ['window','start_step','manifest','sha256']})
    if ds!=allowed_ds:raise ValueError('Unexpected duration metadata change')
    if c!=expected or ds['window']!=480 or ds['start_step']!=prefix or ds['sample_list_sha256']!=parent['duration_sampling']['sample_list_sha256']:
        raise ValueError('Unapproved change beyond world/batch/schedule/output')
    if Path(c['output']).resolve()==Path(parent['output']).resolve():raise ValueError('Independent migration output required')
    return parent


class Schedule:
    def __init__(self,c):
        parent=normalized(c);self.prefix=c['continuation30']['prefix_step'];expected,proof=make_remaining(parent,self.prefix)
        ds=c['duration_sampling'];p=Path(ds['manifest'])
        if sha(p)!=ds['sha256'] or read(p)!=dict(pairs=[list(x) for x in expected],proof=proof):raise ValueError('Continuation schedule differs')
        self.pairs=expected;self.steps=c['training']['steps'];self.seed=c['training']['seed']
        with (Path(c['j_cache'])/'sample_list.csv').open() as f:self.size=sum(1 for _ in csv.DictReader(f))
    def indices(self,size,step,batch,seed,rank=0,world=1):
        if (size,batch,seed,world)!=(self.size,30,self.seed,30) or not 0<=rank<30 or not self.prefix<step<=self.steps:
            raise ValueError('Require30 ranks and valid remaining step')
        return [self.pairs[(step-self.prefix-1)*30+rank]]


def install():
    from p2_whisper_ablation import config as C,training as T,data as D,model as M
    if getattr(C,'_continue30',False):return
    old_validate=C.validate;old_baseline=C.validate_baseline;old_source=C.source_identity;old_schedule=C.Schedule
    def validate(c,formal=True):
        if 'continuation30' not in c:return old_validate(c,formal)
        old_validate(normalized(c),formal);return c
    def baseline(c,ref):
        if 'continuation30' not in c:return old_baseline(c,ref)
        old_baseline(normalized(c),ref);return c
    def source():
        v=old_source()
        for p in [Path(__file__),ROOT/'scripts/p2_whisper_continue30.py']:v[str(p)]=sha(p)
        return v
    class DynamicSchedule:
        def __new__(cls,c):return Schedule(c) if 'continuation30' in c else old_schedule(c)
    C.validate=T.validate=D.validate=M.validate=validate
    C.validate_baseline=T.validate_baseline=baseline
    C.source_identity=T.source_identity=source
    C.Schedule=DynamicSchedule;C._continue30=True
