"""Versioned duration scheduling without changing native cache/source identities."""
from pathlib import Path
import hashlib,json,math,random

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def schedule(rows,steps,batch,seed,policy='duration384',window=384):
    if not rows or steps<1 or batch<1 or window<batch or window%batch:
        raise ValueError('Invalid duration schedule dimensions')
    if policy not in ('random','duration384'):
        raise ValueError('Unknown duration scheduling policy')
    durations=[float(row['requested_duration_s']) for row in rows]
    if any(not math.isfinite(x) or x<=0 for x in durations):
        raise ValueError('Invalid sample duration')
    pairs=[];epoch=0
    while len(pairs)<steps*batch:
        order=list(range(len(rows)));random.Random(seed+epoch).shuffle(order)
        pairs.extend((index,epoch*len(rows)+offset) for offset,index in enumerate(order))
        epoch+=1
    pairs=pairs[:steps*batch]
    if policy=='duration384':
        aligned=[]
        for start in range(0,len(pairs),window):
            block=sorted(pairs[start:start+window],key=lambda pair:durations[pair[0]])
            batches=[block[i:i+batch] for i in range(0,len(block),batch)]
            # Avoid turning length sorting into a short-to-long curriculum.
            random.Random(seed+104729+start).shuffle(batches)
            aligned.extend(pair for group in batches for pair in group)
        pairs=aligned
    return pairs

class Schedule:
    def __init__(self,c):
        spec=c['duration_sampling'];t=c['training'];path=Path(c['guided_view'])/'observations.json'
        if sha(path)!=spec['observations_sha256']:
            raise ValueError('Duration observations changed')
        self.size=len(json.loads(path.read_text()));self.seed=t['seed'];self.batch=t['global_batch']
        if sha(spec['manifest'])!=spec['sha256']:
            raise ValueError('Duration schedule manifest changed')
        manifest=json.loads(Path(spec['manifest']).read_text())
        expected=schedule(json.loads(path.read_text()),t['steps'],self.batch,self.seed,spec['policy'],spec['window'])
        self.pairs=[tuple(p) for p in manifest['pairs']]
        if self.pairs!=expected or manifest['observations_sha256']!=spec['observations_sha256']:
            raise ValueError('Duration schedule does not match exact experiment')

    def indices(self,size,step,batch,seed,rank=0,world=1):
        if (size,batch,seed)!=(self.size,self.batch,self.seed) or world<1 or batch%world or not 0<=rank<world:
            raise ValueError('Duration sampler geometry mismatch')
        if not 1<=step<=len(self.pairs)//batch:
            raise ValueError('Duration sampler step out of range')
        start=(step-1)*batch
        return [self.pairs[start+micro*world+rank] for micro in range(batch//world)]

def install_backend():
    from p2_guided_chain import backend
    if getattr(backend,'_duration_installed',False):return backend
    original_source=backend.source_identity;original_initialize=backend.initialize_stage2
    def source_identity():
        value=original_source()
        for name in ('p2_duration_execution.py','p2_duration_train.py','p2_duration_queue.py'):
            path=ROOT/'scripts'/name;value[str(path)]=sha(path)
        return value
    def initialize(path,model,c,did):
        from p2_matrix16.checkpoint import checked
        parent=checked(path)['identity']['experiment']
        if parent.get('duration_sampling')!=c.get('duration_sampling'):
            raise ValueError('Paired stages use different duration schedules')
        return original_initialize(path,model,c,did)
    backend.source_identity=source_identity;backend.initialize_stage2=initialize
    backend._duration_installed=True
    return backend

def install_sampler(c):
    from p2_listen import training
    selected=Schedule(c);training.indices=selected.indices
    return selected
