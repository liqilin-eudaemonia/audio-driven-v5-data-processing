"""Four-host CPU payload audit; merge the same canonical COMPLETE receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json


def worker(config,rank,world,threads=16):
    import torch
    from p2_whisper_av.cache import Reader
    torch.set_num_threads(2)
    c=read(config);reader=Reader(c['whisper_cache'],source=c['j_cache']);work=reader.work/'distributed_audit'
    if not 0<=rank<world or threads<1:raise ValueError('Invalid CPU audit partition')
    ids=reader.protocol['sample_ids'][rank::world];records=[];started=time.time()
    def check(sid):
        value=reader.load(sid)
        return [sid,value.hidden.shape[1],sha(reader.folder(sid)/'complete.json')]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for offset in range(0,len(ids),256):
            records.extend(pool.map(check,ids[offset:offset+256]))
            atomic_json(work/f'progress_rank{rank:02d}.json',dict(rank=rank,world=world,samples=len(records),
                total=len(ids),elapsed_s=time.time()-started,at=time.time(),success=False))
    result=dict(success=True,rank=rank,world=world,protocol_sha256=sha(reader.work/'protocol.json'),
                code_sha256=sha(__file__),records=records)
    atomic_json(work/f'rank{rank:02d}.json',result)
    print(json.dumps(dict(success=True,rank=rank,samples=len(records),elapsed_s=time.time()-started)),flush=True)


def merge(config,world):
    c=read(config);cache=Path(c['whisper_cache']);p=read(cache/'protocol.json');entries={}
    protocol_sha=sha(cache/'protocol.json')
    for rank in range(world):
        result=read(cache/'distributed_audit'/f'rank{rank:02d}.json')
        if (result.get('success') is not True or result['rank']!=rank or result['world']!=world
                or result['protocol_sha256']!=protocol_sha or result['code_sha256']!=sha(__file__)
                or [row[0] for row in result['records']]!=p['sample_ids'][rank::world]):
            raise ValueError('Incomplete or mismatched distributed audit')
        for sid,tokens,receipt in result['records']:
            if sid in entries:raise ValueError('Duplicate audited sample')
            entries[sid]=(tokens,receipt)
    if len(entries)!=p['samples']:raise ValueError('Missing audited samples')
    h=hashlib.sha256();tokens=0
    for sid in p['sample_ids']:
        n,receipt=entries[sid];tokens+=n;h.update((sid+':'+receipt+'\n').encode())
    result=dict(success=True,samples=p['samples'],tokens=tokens,protocol_sha256=protocol_sha,
                cache_receipts_sha256=h.hexdigest(),audio_source='speaker',intermediate_wav=False)
    atomic_json(cache/'COMPLETE.json',result,immutable=True)
    return result


def run(config,cluster,work):
    from listen_cluster import launch
    c=read(config);cluster=read(cluster);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    nodes=cluster['nodes'];commands=[]
    for rank,node in enumerate(nodes):
        commands.append(['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',
            cluster['python'],str(Path(__file__).resolve()),'worker','--config',str(Path(config).resolve()),
            '--rank',str(rank),'--world',str(len(nodes)),'--threads','16'])
    atomic_json(work/'commands.json',dict(commands=commands,stage='four_host_cpu_audit'))
    atomic_json(work/'state.json',dict(phase='auditing_cache',hosts=len(nodes),threads_per_host=16,at=time.time()))
    launch(cluster,commands,work/('launch_'+str(time.time_ns())))
    result=merge(config,len(nodes));atomic_json(work/'state.json',dict(phase='complete',at=time.time(),**result))
    # Keep the original queue status useful for existing README links.
    q=ROOT/'experiments/p2_j_speaker_whisper_cache_20261104/queue';state=read(q/'state.json')
    state.update(phase='complete',heartbeat=time.time(),samples=result['samples'],cache=c['whisper_cache'],
        cache_complete_sha256=sha(Path(c['whisper_cache'])/'COMPLETE.json'),training_started=False,
        audit_execution='four-hosts-x16-cpu-threads')
    atomic_json(q/'state.json',state);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','merge','run'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=4);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--cluster',type=Path);p.add_argument('--work',type=Path);a=p.parse_args()
    if a.command=='worker':worker(a.config,a.rank,a.world,a.threads)
    elif a.command=='merge':print(json.dumps(merge(a.config,a.world)))
    else:run(a.config,a.cluster,a.work)
