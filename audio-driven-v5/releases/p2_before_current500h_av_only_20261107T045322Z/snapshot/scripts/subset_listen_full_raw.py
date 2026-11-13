"""Freeze an original-pixel crop subset and reuse immutable raw answers by identity."""
import argparse,hashlib,json,math,sys,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import views
from p2_listen.contracts import atomic_json
try:
    import orjson
    loads=orjson.loads
except ImportError:loads=json.loads


def eligible(row,minimum):
    crop=row.get('crop_proposal') or {}
    size=crop.get('size')
    return (row.get('video_transform') is not None and type(size) in (float,int)
            and math.isfinite(size) and size>=minimum)


def filter_queue(source,target,minimum,expected):
    out=target/'queue';(out/'shards').mkdir(parents=True,exist_ok=True)
    if (out/'index.json').exists():raise ValueError('Already indexed')
    def one(path):
        count=Counter();seconds=0.;digest=hashlib.sha256()
        dest=out/'shards'/path.name
        with path.open('rb') as f,dest.open('wb') as g:
            for line in f:
                digest.update(line);row=loads(line);count['seen']+=1
                if not eligible(row,minimum):count['outside_scope']+=1;continue
                g.write(line);count['queued']+=1;count[row['admission_pool']]+=1
                count['views_per_branch']+=len(views(row['duration_s']));seconds+=row['duration_s']
        return path.stem,count,seconds,digest.hexdigest()
    began=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,sorted((source/'queue/shards').glob('*.jsonl'))))
    counts=Counter()
    for _,c,_,_ in results:counts.update(c)
    if counts['queued']!=expected:raise ValueError(f'Expected {expected}, got {counts["queued"]}')
    counts['crop_unavailable']=0
    record=dict(version='listen-full-raw-crop-subset-v1',complete=True,counts=counts,
        hours=sum(x[2] for x in results)/3600,shards={k:c['queued'] for k,c,_,_ in results},
        source_run=str(source),source_shard_sha256={k:h for k,_,_,h in results},
        scope=dict(field='crop_proposal.size',minimum_native_pixels=minimum,uses_unrounded_size=True,
                   preserves_source_segmentation=True,preserves_onset_flags=True),seconds=time.time()-began)
    atomic_json(out/'index.json',record);print(json.dumps(record['counts']),flush=True)


def import_raw(source,target,minimum):
    if not (source/'RAW_FROZEN').exists():raise ValueError('Wait until all source workers have exited before raw reuse')
    index=json.loads((target/'queue/index.json').read_text());expected=index['shards']
    report={};all_selected=set()
    for branch in ['visual','audio','asr']:
        for name in ['results','state','locks']:(target/'raw'/branch/name).mkdir(parents=True,exist_ok=True)
        totals=Counter()
        for shard in expected:
            path=source/'raw'/branch/'results'/(shard+'.jsonl')
            if not path.exists():continue
            parents=[loads(l) for l in (target/'queue/shards'/(shard+'.jsonl')).read_bytes().splitlines() if l]
            selected={r['sample_id']:r for r in parents};all_selected.update(selected)
            latest={};copied=0;dest=target/'raw'/branch/'results'/path.name
            if dest.exists():raise ValueError('Do not overwrite imported raw results')
            with path.open('rb') as f,dest.open('wb') as out:
                for line in f:
                    r=loads(line);sid=r.get('sample_id')
                    if sid not in selected:continue
                    if r.get('source',{}).get('source_line_sha256')!=selected[sid]['source_line_sha256']:
                        raise ValueError('Changed sample identity during raw import')
                    out.write(line);copied+=1
                    if r.get('record_kind')=='result':latest[r['request_id']]=r['status']
            counts=Counter(latest.values());totals.update(counts)
            done=all((r['sample_id']+'@input' in latest or all(r['sample_id']+'@'+name in latest
                    for name,_,_ in views(r['duration_s']))) for r in parents)
            atomic_json(target/'raw'/branch/'state'/(shard+'.json'),dict(shard=shard,worker='imported',done=done,
                counts=counts,requests=len(latest),parents=len(parents),heartbeat=time.time(),source_raw_file=str(path)))
        report[branch]=dict(counts=totals,requests=sum(totals.values()))
    repairs=source/'raw/asr_alignment_repairs/results.jsonl';repaired=0
    if repairs.exists():
        folder=target/'raw/asr_alignment_repairs';folder.mkdir(parents=True,exist_ok=True)
        with repairs.open('rb') as f,(folder/'results.jsonl').open('wb') as out:
            for line in f:
                r=loads(line)
                if r['request_id'].rsplit('@',1)[0] in all_selected:
                    out.write(line);repaired+=r['status']=='complete'
        atomic_json(folder/'status.json',dict(complete=repaired,done=True,imported_from=str(repairs)))
    atomic_json(target/'reuse_report.json',dict(source_run=str(source),branches=report,
        raw_records_preserved_byte_for_byte=True,alignment_repairs=repaired,scope=index['scope']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--minimum',type=float,default=384);p.add_argument('--expected',type=int,default=722706)
    p.add_argument('--phase',choices=['queue','reuse'],required=True);a=p.parse_args()
    source,target=Path(a.source).resolve(),Path(a.target).resolve()
    if a.phase=='queue':filter_queue(source,target,a.minimum,a.expected)
    else:import_raw(source,target,a.minimum)
