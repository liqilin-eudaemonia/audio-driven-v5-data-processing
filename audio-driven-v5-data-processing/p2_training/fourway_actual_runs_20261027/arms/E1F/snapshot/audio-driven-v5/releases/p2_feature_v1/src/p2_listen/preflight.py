"""One full audit, persistent dependency receipt, one metadata scan per launch.

The shared-filesystem contract is explicit. Per-sample training cache SHA256 checks
remain in the original loader. Stat fingerprints are a controlled-storage change
indicator, not a cryptographic proof against hostile metadata preservation.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .contracts import ROOT, EVENTS, atomic_json, digest, path_at, read, sha, observation, target, audit_splits

SCHEMA='p2-shared-preflight-v1'


def file_state(path):
    p=Path(path);s=p.stat()
    if not stat.S_ISREG(s.st_mode):raise ValueError(f'Expected file: {p}')
    state=dict(resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,
               ctime_ns=s.st_ctime_ns,inode=s.st_ino)
    # The mounted filesystem can round timestamps to whole seconds. Always hash
    # JSON contracts/labels and small files (including event tensors); these carry
    # class statistics. Large AV payloads retain SHA verification on actual reads.
    if p.suffix=='.json' or s.st_size<=65536:state['sha256']=sha(p)
    return state


def bound_json(path):
    before=file_state(path);data=Path(path).read_bytes();after=file_state(path)
    if before!=after:raise ValueError(f'File changed while reading: {path}')
    return json.loads(data),hashlib.sha256(data).hexdigest(),after


def metadata(c):
    from .config import source_identity,training_cache_parts,feature_loss_enabled
    from .features import weights_identity
    root=path_at(c['output'])
    op=root/'manifests/observations.json';tp=root/'manifests/targets.json'
    raw_obs,oh,ostate=bound_json(op);raw_targets,th,tstate=bound_json(tp)
    rows=[observation(o,verify=False) for o in raw_obs if o['experiment_split']=='train']
    if not rows:raise ValueError('No admitted training observations')
    audit_splits(raw_obs)
    raw={r['sample_id']:r for r in raw_targets}
    targets={o['sample_id']:target(raw[o['sample_id']],o,verify=False) for o in rows}
    parts=training_cache_parts(c)
    views={name:str((root/name).resolve()) for name in ['labels/canonical',*[f'cache/{p}' for p in parts]]}
    config={k:v for k,v in c.items() if k!='output'}
    # Original access paths must stay in the ledger, so changed symlink targets invalidate it.
    binding=dict(schema=SCHEMA,config=config,manifests={str(op):oh,str(tp):th},views=views,
                 source=source_identity(c),assets={k:weights_identity(c,k) for k in ('omni','ltx','gemma')},
                 feature_loss_enabled=feature_loss_enabled(c))
    return dict(rows=rows,targets=targets,binding=binding,manifest_states={str(op):ostate,str(tp):tstate})


def dependencies(c,meta):
    from .config import training_cache_parts
    from .features import folder
    paths=list(meta['manifest_states'])+list(meta['binding']['source'])
    for files in meta['binding']['assets'].values():paths.extend(files)
    for o in meta['rows']:
        t=meta['targets'][o['sample_id']]
        paths.extend([o['speaker_audio'],o['listener_image'],t['target_video'],t['target_audio'],
                      str(path_at(c['output'])/'labels/canonical'/(o['sample_id']+'.json'))])
        for part in training_cache_parts(c):
            if part=='text':continue
            for name in ('complete.json','payload.pt'):paths.append(str(folder(c,part,o['sample_id'])/name))
    for name in ('complete.json','payload.pt'):paths.append(str(folder(c,'text')/name))
    return sorted(set(map(str,paths)))


def ordered_io(function,items):
    # Bound both filesystem concurrency and queued work. Results remain in input
    # order, especially event labels used by the original floating-point sums.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(items),64):
            yield from pool.map(function,items[start:start+64])


def snapshot(paths):return dict(zip(paths,ordered_io(file_state,paths)))


def stable_snapshot(paths):
    # Some mounts expose only second-resolution timestamps. Before the audit,
    # let large, unhashed files age out of that ambiguity window. Any later write
    # changes ctime even if a writer restores mtime. JSON/small files are hashed.
    result=snapshot(paths)
    recent=[max(v['mtime_ns'],v['ctime_ns'])/1e9 for v in result.values() if 'sha256' not in v]
    delay=max(recent,default=0)+2-time.time()
    if delay>0:
        if delay>5:raise ValueError('Future dependency timestamps; cannot establish a stable audit')
        time.sleep(delay)
        result=snapshot(paths)
        if any('sha256' not in v and max(v['mtime_ns'],v['ctime_ns'])/1e9+2>time.time() for v in result.values()):
            raise ValueError('Dependencies are still being written; retry after writers finish')
    return result


def unchanged(expected):
    try:return snapshot(list(expected))==expected
    except (OSError,ValueError):return False


def full_audit(c,meta):
    from .training_data import load_training_sample
    from .features import load_text
    from .annotations import class_statistics
    before=stable_snapshot(dependencies(c,meta))
    if any(before[p]!=s for p,s in meta['manifest_states'].items()):raise ValueError('Manifest changed before full audit')
    labels=[];signatures=[];canonical=[]
    def inspect_sample(o):
        t=meta['targets'][o['sample_id']]
        observation(o);target(t,o)  # Preserve full original media SHA checks, once globally.
        sample=load_training_sample(c,o,t)
        label=sample[4];del sample
        original=read(path_at(c['output'])/'labels/canonical'/(o['sample_id']+'.json'))
        # The unchanged admission policy reads only these provenance fields.
        provenance=dict(provenance={k:original['provenance'].get(k) for k in ('debug_only','formal_training_admitted')})
        return label,digest(original),provenance
    for label,signature,provenance in ordered_io(inspect_sample,meta['rows']):
        labels.append(label);signatures.append(signature);canonical.append(provenance)
    load_text(c,'cpu')
    statistics=class_statistics(labels,c['events']['pos_weight_cap'],c['events']['K_train'])
    if not unchanged(before):raise ValueError('Dependency changed during full audit; no success receipt published')
    if metadata(c)['binding']!=meta['binding']:raise ValueError('Audit binding changed during full audit')
    return dict(binding=meta['binding'],dependencies=before,result=dict(signatures=signatures,canonical=canonical,statistics=statistics))


def get_audit(c,meta,cache_dir=None):
    begin=time.monotonic();key=digest(meta['binding'])
    cache=Path(cache_dir or os.environ.get('P2_PREFLIGHT_CACHE',path_at(c['output'])/'preflight'))
    cache.mkdir(parents=True,exist_ok=True)
    receipt=cache/(key+'.json');marker=cache/(key+'.complete.json')
    with (cache/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        invalidated=False
        if marker.exists():
            stamp=read(marker)
            if stamp.get('schema')!=SCHEMA or stamp.get('binding')!=key or sha(receipt)!=stamp['sha256']:
                raise ValueError('Preflight receipt integrity check failed')
            saved=read(receipt)
            if saved['binding']!=meta['binding']:raise ValueError('Preflight receipt binding differs')
            if unchanged(saved['dependencies']):
                return saved['result'],dict(mode='reused',full_audits=0,dependencies=len(saved['dependencies']),
                    seconds=time.monotonic()-begin,receipt=str(receipt),receipt_sha256=stamp['sha256'])
            invalidated=True
        saved=full_audit(c,meta)
        atomic_json(receipt,saved)
        stamp=dict(schema=SCHEMA,binding=key,sha256=sha(receipt))
        atomic_json(marker,stamp)
        return saved['result'],dict(mode='rebuilt' if invalidated else 'full',full_audits=1,
            dependencies=len(saved['dependencies']),seconds=time.monotonic()-begin,receipt=str(receipt),receipt_sha256=stamp['sha256'])


def wait_until(check,timeout):
    begin=time.monotonic()
    while True:
        result=check()
        if result:return result
        if time.monotonic()-begin>timeout:raise TimeoutError('Shared preflight timed out; inspect rank requests/results')
        time.sleep(.2)


def distributed_audit(c,rank,world,session,timeout):
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,128}',session):raise ValueError('P2_PREFLIGHT_SESSION must be a fresh shared 8–128 character token')
    directory=path_at(c['output'])/'preflight_sessions'/session
    directory.mkdir(parents=True,exist_ok=True)
    request_path=directory/f'request_{rank}.json'
    # O_EXCL ownership prevents both duplicate ranks and reuse of stale success tickets.
    reservation=directory/f'rank_{rank}.reservation'
    with reservation.open('x') as f:f.write(str(os.getpid()))
    nonce=uuid.uuid4().hex;meta=None
    try:
        meta=metadata(c);request=dict(rank=rank,world=world,nonce=nonce,binding=digest(meta['binding']))
    except Exception as error:
        request=dict(rank=rank,world=world,nonce=nonce,error=f'{type(error).__name__}: {error}')
    atomic_json(request_path,request)
    result_path=directory/'result.json'
    if rank==0:
        try:
            wait_until(lambda:all((directory/f'request_{r}.json').exists() for r in range(world)),timeout)
            requests=[read(directory/f'request_{r}.json') for r in range(world)]
            if any(r.get('error') for r in requests):raise ValueError('Rank preflight failure: '+str(requests))
            if any(r['rank']!=i or r['world']!=world or r['binding']!=requests[0]['binding'] for i,r in enumerate(requests)):
                raise ValueError('Ranks disagree on preflight config/data/source/assets')
            result,report=get_audit(c,meta)
            response=dict(status='success',requests=requests,result=result,report=report)
        except Exception as error:
            response=dict(status='failed',error=f'{type(error).__name__}: {error}')
        atomic_json(result_path,response)
    wait_until(result_path.exists,timeout)
    response=read(result_path)
    if response['status']!='success':raise RuntimeError('Shared preflight failed: '+response['error'])
    if response['requests'][rank]!=request:raise ValueError('Stale/mismatched preflight response')
    atomic_json(directory/f'ack_{rank}.json',dict(nonce=nonce,rank=rank,world=world,binding=request['binding']))
    if rank==0:wait_until(lambda:all((directory/f'ack_{r}.json').exists() for r in range(world)),timeout)
    # Each rank retains its own verified manifest decode; statistics are rank0's exact result.
    return dict(rows=meta['rows'],targets=meta['targets'],**response['result']),response['report'],directory


def prepare_training(c):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if world<1 or not 0<=rank<world:raise ValueError('Invalid preflight rank/world')
    timeout=float(os.environ.get('P2_PREFLIGHT_TIMEOUT_SECONDS','3600'))
    if not 0<timeout<float('inf'):raise ValueError('Invalid preflight timeout')
    begin=time.monotonic()
    if world>1:
        session=os.environ.get('P2_PREFLIGHT_SESSION','')
        if not session:raise ValueError('Distributed fast preflight requires a new shared P2_PREFLIGHT_SESSION token')
        result,report,directory=distributed_audit(c,rank,world,session,timeout)
    else:
        meta=metadata(c);summary,report=get_audit(c,meta)
        result=dict(rows=meta['rows'],targets=meta['targets'],**summary)
        directory=path_at(c['output'])/'preflight_sessions'/uuid.uuid4().hex
    record=dict(schema=SCHEMA,rank=rank,world=world,total_seconds=time.monotonic()-begin,
                global_audit=report,local_full_audits=report['full_audits'] if rank==0 else 0,
                samples=len(result['rows']),statistics_sha256=digest(result['statistics']))
    atomic_json(directory/f'rank_{rank}_report.json',record)
    print('P2_PREFLIGHT '+json.dumps(record),flush=True)
    return result
