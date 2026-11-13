"""Immutable observation-only protocol. Preparation never opens labels or GT."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json,observation,safe_id
from .planning import SYSTEM,DEFAULTS,parse_schedule
from . import VERSION

@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

def code_identity():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/p2_autonomous_eval.py']
    for folder in ['p2_plan_cache','p2_guided_prompt_en']:
        paths+=list((ROOT/'src'/folder).glob('*.py'))
    paths+=[ROOT/'src/p3_listen/features.py',ROOT/'src/p3_listen/prompts.py']
    return {str(p):sha(p) for p in sorted(set(paths))}

def prepare(config,observations,job_source,work):
    from p2_listen.config import load
    work=Path(work).resolve();paths=[Path(p).resolve() for p in (config,observations,job_source)]
    if any(p.is_relative_to(work) for p in paths):raise ValueError('Use a separate output directory')
    c=load(paths[0]);obs=[observation(o,paths[1].parent) for o in read(paths[1])]
    ids=[o['sample_id'] for o in obs]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Empty/duplicate observations')
    if any(o['geometry']['height']!=512 or o['geometry']['width']!=512 for o in obs):raise ValueError('Expected512x512')
    sources={str(p):sha(p) for p in paths};jobs=[];seen=set()
    allowed={'family','arm','split','sample_id','seed','checkpoint','run','restoration','output'}
    for row in read(paths[2]):
        if set(row)-allowed:raise ValueError('Unexpected job fields; do not pass labels/GT')
        if row['sample_id'] not in ids:raise ValueError('Jobs include unselected samples')
        if row['family'] not in ('legacy','matrix') or row['seed'] not in (137,251):raise ValueError('Invalid job family/seed')
        arm='autonomous_'+safe_id(row['arm']);sid=safe_id(row['sample_id']);key=(arm,sid,row['seed'])
        if key in seen:raise ValueError('Duplicate job')
        seen.add(key);j=dict(row,arm=arm,source_arm=row['arm'],output=str(work/'outputs'/arm/(sid+'__seed'+str(row['seed']))))
        j['work']=str(work);jobs.append(j)
    arms=sorted({j['arm'] for j in jobs})
    if not arms or seen!={(a,s,seed) for a in arms for s in ids for seed in (137,251)}:raise ValueError('Incomplete paired job matrix')
    p=dict(schema=VERSION,config=c,observations=obs,sources=sources,code=code_identity(),planner=DEFAULTS,
           planner_system=SYSTEM,plan_source='qwen_autonomous',target_annotation_access=False,gt_media_access=False,
           shared_plan_across_arms_and_generation_seeds=True,hidden_scope='all_input_and_plan_tokens',jobs=jobs,
           metric_adapter_sha256=sha(ROOT/'experiments/p2_reintegrated_final_eval_20261028/eval_metrics.py'))
    with lock(work.parent/(work.name+'.prepare.lock')):
        atomic_json(work/'protocol.json',p,immutable=True);atomic_json(work/'jobs.json',jobs,immutable=True)
        for j in jobs:atomic_json(work/'jobs'/(digest(j)+'.json'),j,immutable=True)
    return dict(samples=len(obs),jobs=len(jobs),arms=arms,work=str(work),gpu_started=False)

def open_protocol(work):
    p=read(Path(work)/'protocol.json')
    if p['schema']!=VERSION or p['code']!=code_identity() or p['planner']!=DEFAULTS or p['planner_system']!=SYSTEM:
        raise ValueError('Autonomous protocol/code changed; prepare a new work directory')
    if p['target_annotation_access'] is not False or p['gt_media_access'] is not False:raise ValueError('Not source-only')
    for name,h in p['sources'].items():
        if sha(name)!=h:raise ValueError('Prepared input/config/jobs changed: '+name)
    return p

def selected(p,shard_index=0,shards=1):
    if type(shard_index) is not int or type(shards) is not int or not 0<=shard_index<shards:raise ValueError('Invalid shard')
    return sorted(p['observations'],key=lambda o:o['sample_id'])[shard_index::shards]

def plan_identity(p,obs,assets):
    return dict(protocol_sha256=digest(p),observation_sha256=digest(obs),planner_assets=assets,plan_source='qwen_autonomous')

def read_plan(work,p,sid,expected=None):
    safe_id(sid);folder=Path(work)/'plans'/sid;marker=read(folder/'complete.json')
    if marker['protocol_sha256']!=digest(p):raise ValueError('Planner protocol mismatch')
    if expected is not None and marker['identity']!=expected:raise ValueError('Planner model/input identity changed')
    for name,h in marker['files'].items():
        if sha(folder/name)!=h:raise ValueError('Planner artifact changed')
    result=read(folder/'plan.json');obs=next(o for o in p['observations'] if o['sample_id']==sid)
    rebuilt=parse_schedule(result['raw'],sid,obs['requested_duration_s'],p['planner']['max_events'])
    if result!=rebuilt:raise ValueError('Plan/text no longer matches raw model output')
    return result
