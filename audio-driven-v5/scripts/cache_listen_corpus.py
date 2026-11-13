"""CPU export shards and GPU feature shards, with durable per-shard completion seals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import fcntl
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
os.environ['PATH']=str(ROOT/'scripts/media_bin')+os.pathsep+os.environ['PATH']
sys.path.insert(0,str(ROOT/'src'))


def configure_media_export(media):
    """Keep the final source frame when CFR rounding reaches the source EOF.

    Production-only adapter: historical S0 source/checkpoint hashes stay intact.
    FFmpeg's default eof_action=round can discard the last frame of a valid
    window. pass rounds the EOF boundary upwards; frame-count/audio checks
    still reject genuinely short sources. No padding filter is used.
    """
    original_command=media.command
    def command(args):
        args=list(args)
        if '-vf' in args:
            i=args.index('-vf')+1
            args[i]=re.sub(r'fps=(\d+)(?=,|$)',r'fps=\1:eof_action=pass',args[i])
        return original_command(args)
    media.command=command


def bounded_results(pool, function, rows, capacity):
    """Avoid submitting all 272k rows/futures at once; yield completed work."""
    source=iter(rows);pending=set()
    for _ in range(capacity):
        row=next(source,None)
        if row is None:break
        pending.add(pool.submit(function,row))
    while pending:
        ready,pending=wait(pending,return_when=FIRST_COMPLETED)
        for future in ready:
            yield future.result()
            row=next(source,None)
            if row is not None:pending.add(pool.submit(function,row))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('export','cache','seal'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/p3_listen_300h.json')
    p.add_argument('--node',type=int,default=0);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--limit',type=int,default=0,help='Engineering checks only; cannot seal production')
    a=p.parse_args()
    from p3_listen.common import config,resolve,read,write,write_once,digest,sha,source_contract
    from p3_listen.media import export,verify_media
    cfg=config(a.config);root=resolve(cfg['output']);manifest=read(root/'windows.json')
    sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
    rows=[r for r in manifest['windows'] if r['split'] in ('train','dev')]
    if a.limit:rows=rows[:a.limit]
    if a.command=='export':
        from p3_listen import media
        from listen_fast_media import export
        configure_media_export(media)
        from functools import lru_cache
        original_sha=media.sha
        @lru_cache(maxsize=40000)
        def normalized_sha(path,size,mtime):return original_sha(path)
        def source_sha(path):
            path=Path(path)
            if 'normalized_audio' in path.parts:
                stat=path.stat();return normalized_sha(str(path),stat.st_size,stat.st_mtime_ns)
            return original_sha(path)
        media.sha=source_sha
        selected=rows[a.node::3]+[r for i,r in enumerate(rows) if i%3!=a.node]
        (root/'export_locks').mkdir(exist_ok=True);begin=time.monotonic()
        def shared_export(row):
            with (root/'export_locks'/(row['sample_id']+'.lock')).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return
                marker=media.paths(root,row)['marker']
                if marker.exists():
                    if read(marker)['row_sha256']!=digest(row):raise ValueError('Export contract changed')
                    return
                error_path=root/'export_errors'/(row['sample_id']+'.json')
                for attempt in range(3):
                    try:
                        export(root,row)
                        error_path.unlink(missing_ok=True)
                        return
                    except Exception as e:
                        if attempt<2:
                            time.sleep(1)
                            continue
                        write(error_path,dict(sample_id=row['sample_id'],node=a.node,
                            error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()))
                        print(json.dumps(dict(export_failed=row['sample_id'],error=str(e))),flush=True)
                        return row['sample_id']
        failed=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i,error in enumerate(bounded_results(pool,shared_export,selected,2*a.workers),1):
                if error:failed.append(error)
                if i%100==0 or i==len(selected):
                    report=dict(node=a.node,visited=i,total=len(selected),failed=len(failed),shared_work_queue=True,seconds=time.monotonic()-begin)
                    write(root/f'export_progress_{a.node}.json',report);print(report,flush=True)
        # Another node may have successfully retried a failed row meanwhile.
        unresolved=[sid for sid in failed if (root/'export_errors'/(sid+'.json')).exists()]
        if unresolved:raise RuntimeError(f'{len(unresolved)} exports failed; see {root / "export_errors"}')
        write(root/f'export_completed_{a.node}.json',dict(manifest_sha256=sha(root/'windows.json'),count=len(selected),limit=a.limit))
    elif a.command=='cache':
        rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
        visible=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8)))).split(',')
        os.environ['CUDA_VISIBLE_DEVICES']=visible[local]
        os.environ['TOKENIZERS_PARALLELISM']='false'
        selected=rows[rank::world]
        from p3_listen import features
        # Each rank owns non-overlapping sample IDs. Fixed text/global contracts are prepared once.
        features.selection=lambda cfg,args:(manifest,selected)
        a.part='all'
        features.cache(cfg,a)
        # All outputs are re-opened and validated before the shard can be sealed.
        h=hashlib.sha256()
        for i,row in enumerate(selected,1):
            features.load_native(root,row,cfg,targets=True)
            features.load_av(root,row,cfg,'cpu')
            folder=features.cache_dir(root,row)
            h.update(digest([row['sample_id'],sha(folder/'native.json'),sha(folder/'vae.json')]).encode())
        write(root/'cache_shards'/f'rank_{rank:02d}.json',dict(rank=rank,world_size=world,
            manifest_sha256=sha(root/'windows.json'),config_sha256=digest(cfg),count=len(selected),
            sample_ids_sha256=digest([r['sample_id'] for r in selected]),marker_digest=h.hexdigest(),
            verified_file_hashes=True,source=source_contract(cfg),limit=a.limit))
    else:
        if a.limit:raise ValueError('Cannot seal a partial corpus')
        from p3_listen import features
        markers=[]
        for rank in range(24):
            value=read(root/'cache_shards'/f'rank_{rank:02d}.json')
            expected=rows[rank::24]
            if (value['world_size']!=24 or value['limit'] or value['count']!=len(expected)
                or value['sample_ids_sha256']!=digest([r['sample_id'] for r in expected])
                or value['manifest_sha256']!=sha(root/'windows.json') or value['config_sha256']!=digest(cfg)
                or value['source']!=source_contract(cfg) or not value['verified_file_hashes']):
                raise ValueError('Invalid shard seal: '+str(rank))
            markers.append(value)
        features.load_text(root,cfg,'cpu');features.load_text(root,cfg,'cpu',negative=True)
        if manifest['unique_seconds']['train']<300*3600:raise ValueError('Insufficient training hours')
        write_once(root/'data_ready.json',dict(manifest_sha256=sha(root/'windows.json'),config_sha256=digest(cfg),
            unique_hours={k:v/3600 for k,v in manifest['unique_seconds'].items()},sample_count=len(rows),
            world_size=24,shards=markers,all_files_verified=True,formal_training_allowed=True))


if __name__=='__main__':main()
