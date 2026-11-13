"""Opt-in J-cache duration scheduling with an unchanged consumed prefix."""
import csv
import random
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha
from p2_duration_execution import schedule


def make_pairs(rows,steps,batch,seed,start_step,policy,window=512):
    if policy not in ('random','duration') or not 0<=start_step<=steps or window<batch or window%batch:
        raise ValueError('Invalid duration plan')
    pairs=schedule(rows,steps,batch,seed,'random',window)
    if policy=='random':return pairs
    start=start_step*batch;result=pairs[:start]
    for offset in range(start,len(pairs),window):
        block=sorted(pairs[offset:offset+window],key=lambda p:rows[p[0]]['requested_duration_s'])
        batches=[block[i:i+batch] for i in range(0,len(block),batch)]
        random.Random(seed+104729+offset).shuffle(batches)
        result.extend(pair for group in batches for pair in group)
    return result


class Schedule:
    def __init__(self,c):
        spec=c['duration_sampling'];t=c['training'];path=Path(c['j_cache'])/'sample_list.csv'
        if sha(path)!=spec['sample_list_sha256'] or sha(spec['manifest'])!=spec['sha256']:
            raise ValueError('Duration plan/source changed')
        with path.open() as stream:source=list(csv.DictReader(stream))
        rows=[dict(requested_duration_s=float(r['duration_s'])) for r in source]
        # Dataset order is frozen in the same cache manifest and label database.
        if [r['sample_id'] for r in source]!=read(Path(c['j_cache'])/'manifest.json')['splits']['train']:
            raise ValueError('Duration sample order differs')
        plan=read(spec['manifest']);expected=make_pairs(rows,t['steps'],t['global_batch'],t['seed'],
            spec['start_step'],spec['policy'],spec['window'])
        self.pairs=[tuple(p) for p in plan['pairs']]
        if self.pairs!=expected or plan['sample_list_sha256']!=spec['sample_list_sha256']:
            raise ValueError('Duration plan is not the deterministic schedule')
        self.size=len(rows);self.batch=t['global_batch'];self.seed=t['seed']

    def indices(self,size,step,batch,seed,rank=0,world=1):
        if (size,batch,seed)!=(self.size,self.batch,self.seed) or world<1 or batch%world or not 0<=rank<world:
            raise ValueError('Duration sampler geometry differs')
        if not 1<=step<=len(self.pairs)//batch:raise ValueError('Duration sampler step outside budget')
        start=(step-1)*batch
        return [self.pairs[start+micro*world+rank] for micro in range(batch//world)]


def install_backend():
    from p2_clip_events import backend
    if getattr(backend,'_j_duration_installed',False):return backend
    original=backend.source_identity
    def source_identity():
        value=original()
        for name in ('p2_duration_execution.py','p2_j_duration_execution.py','p2_j_duration_train.py'):
            path=ROOT/'scripts'/name;value[str(path)]=sha(path)
        return value
    backend.source_identity=source_identity
    backend._j_duration_installed=True
    return backend


def install_sampler(c):
    from p2_listen import training
    selected=Schedule(c);training.indices=selected.indices
    return selected
