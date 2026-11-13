#!/usr/bin/env python3
"""Opt-in survivor-only routing over frozen native feature code/shared cache."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import IntegrityError,canonical,lock,read,save,sha
from prepare_glare_prosody_subset import OLD,PARENT,protocol,require


def verify_export(export):
    seal=read(export/'SEAL.json')
    for name,digest in seal['files'].items():
        require(Path(name).name==name and sha(export/name)==digest,'Export seal mismatch')
    return seal


def validate_work_target(work, plan, export, cache_root):
    work=work.resolve()
    for protected in (OLD,PARENT,plan,export,cache_root):
        protected=Path(protected).resolve()
        require(work!=protected and work not in protected.parents and protected not in work.parents,
                'Feature work must be separate from protected data/cache/plan directories')
    if work.exists() and not (work/'scope.json').exists():
        require(not [p for p in work.iterdir() if p.name!='prepare.lock'],
                'Refuse an existing nonempty workspace without this tool scope.json')


def prepare(args):
    p=protocol(args.plan); export=args.export.resolve(); seal=verify_export(export)
    validate_work_target(args.work,args.plan,export,p['cache_root'])
    require(seal['protocol_sha256']==sha(args.plan/'protocol.json'),'Export belongs to another plan')
    summary=read(export/'SUMMARY.json')
    require(args.allow_partial or not summary['counts'].get('pending',0),'Pending pairs remain: complete extraction or explicitly use --allow-partial for a pilot')
    jobs=sqlite3.connect(PARENT/'jobs.sqlite');jobs.row_factory=sqlite3.Row
    by_sid={r['sid']:dict(r) for r in jobs.execute('SELECT * FROM jobs')};jobs.close()
    accepted=[]
    from glare_prosody_subset import ChunkCache
    from prepare_glare_prosody_subset import connect
    cache=ChunkCache(p['cache_root'],p['extractor'])
    with lock(cache.root/'mutation.lock',blocking=False),connect(args.plan) as db:
        with (export/'retained_all.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);sid=row['sample_id']; job=by_sid[sid]
                pair=json.loads(db.execute('SELECT data FROM pairs WHERE sid=?',(sid,)).fetchone()[0])
                track=json.loads(db.execute('SELECT data FROM tracks WHERE track_id=?',(pair['track_id'],)).fetchone()[0])
                require(pair['row_sha']==job['row_sha'] and pair['split']==job['split'],'Parent selection changed')
                if row['glare_prosody'].get('policy')=='official':
                    condition=row['glare_prosody'];receipt=read(condition['intensity_receipt'])
                    require(receipt['identity']['pair']==pair and receipt['identity']['source_contract_id']==cache.contract_id,
                            'Official condition belongs to another pair/source')
                    require(sha(condition['intensity_receipt'])==condition['intensity_receipt_sha256'] and
                            sha(condition['intensity'])==receipt['payload_sha256'],'Official condition payload changed')
                else:
                    decision=cache.admit_pair(pair,track,p['selection_sha256'])
                    require(decision['decision']=='keep' and decision['intensity_receipt_sha256']==row['glare_prosody']['intensity_receipt_sha256'],'Prosody acceptance changed')
                accepted.append(job['ordinal'])
    require(accepted and len(accepted)==len(set(accepted))==summary['counts']['keep'],'Empty/duplicate/incomplete retained list')
    scope=dict(schema='glare-prosody-survivor-routing-v1',plan=str(args.plan.resolve()),export=str(export),
        export_seal_sha256=sha(export/'SEAL.json'),selection_sha256=p['selection_sha256'],
        parent_protocol_sha256=sha(OLD/'protocol.json'),groups_sha256=sha(OLD/'groups.sqlite'),
        parent_jobs_sha256=sha(PARENT/'jobs.sqlite'),ordinals=sorted(accepted),pairs=len(accepted),
        original_cache=str((OLD/'cache').resolve()),full_glare_training_admitted=False,
        code_sha256=sha(__file__))
    args.work.mkdir(parents=True,exist_ok=True)
    with lock(args.work/'prepare.lock'):
        if (args.work/'scope.json').exists():
            require(read(args.work/'scope.json')==scope,'Immutable feature workspace has another scope');return
        require(not [p for p in args.work.iterdir() if p.name!='prepare.lock'],
                'Workspace became nonempty; refuse overwriting protocol or links')
        shutil.copyfile(OLD/'protocol.json',args.work/'protocol.json')
        for name in ('groups.sqlite','cache'):
            path=args.work/name;target=(OLD/name).resolve()
            if path.is_symlink():require(path.resolve()==target,'Wrong shared link')
            else:path.symlink_to(target,target_is_directory=name=='cache')
        acceptance=args.work/'gpu7_only/lia_fused_fp32_acceptance/RESULT.json'
        acceptance.parent.mkdir(parents=True,exist_ok=True)
        original=(OLD/'gpu7_only/lia_fused_fp32_acceptance/RESULT.json').resolve()
        if not acceptance.exists():acceptance.symlink_to(original)
        for folder in ('workers','errors'): (args.work/folder).mkdir(exist_ok=True)
        save(args.work/'scope.json',scope)
    print(json.dumps(dict(work=str(args.work),pairs=len(accepted),running=False)))


def filtered_groups(groups,accepted,durations):
    result=[]
    for original in groups:
        group=dict(original);ordinals=[o for o in json.loads(group['ordinals']) if o in accepted]
        if ordinals:
            group.update(ordinals=json.dumps(ordinals),pairs=len(ordinals),seconds=sum(durations[o] for o in ordinals))
            result.append(group)
    return result


def run(args):
    scope=read(args.work/'scope.json')
    require(scope['code_sha256']==sha(__file__),'Routing code changed; prepare a new workspace')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='7','Only GPU7 is authorized')
    verify_export(Path(scope['export']))
    require(sha(Path(scope['export'])/'SEAL.json')==scope['export_seal_sha256'],'Export changed')
    for path,digest in ((OLD/'protocol.json',scope['parent_protocol_sha256']),
                        (OLD/'groups.sqlite',scope['groups_sha256']),(PARENT/'jobs.sqlite',scope['parent_jobs_sha256'])):
        require(sha(path)==digest,'Frozen parent changed')
    # Original supervisors/workers must all be absent. Never stop or clear their markers here.
    with ExitStack() as locks:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock'):
            locks.enter_context(lock(path,blocking=False))
        require(scope['original_cache']==str((OLD/'cache').resolve()),'Scope points outside original canonical cache')
        jobs=sqlite3.connect(PARENT/'jobs.sqlite');jobs.row_factory=sqlite3.Row
        indexed=[dict(j) for j in jobs.execute('SELECT ordinal,sid,duration FROM jobs')];jobs.close()
        lookup={j['sid']:j for j in indexed};expected=[]
        with (Path(scope['export'])/'retained_all.jsonl').open() as stream:
            for line in stream:expected.append(lookup[json.loads(line)['sample_id']]['ordinal'])
        require(sorted(expected)==scope['ordinals'] and len(expected)==len(set(expected))==scope['pairs'] and bool(expected),
                'Scope ordinals differ from sealed survivor manifest')
        import glare_visual_runtime as runtime
        pipeline=runtime.pipeline;pipeline.WORK=args.work.resolve()
        accepted=set(scope['ordinals'])
        durations={j['ordinal']:j['duration'] for j in indexed}
        class FilteredConnection(sqlite3.Connection):
            def execute(self,sql,parameters=()):
                if sql=='SELECT * FROM groups WHERE worker=? ORDER BY gid':
                    return filtered_groups(super().execute('SELECT * FROM groups ORDER BY gid'),accepted,durations)
                return super().execute(sql,parameters)
        def connect(database,*a,**kw):
            if str(database).split('?')[0].endswith('/groups.sqlite'):kw['factory']=FilteredConnection
            return sqlite3.connect(database,*a,**kw)
        pipeline.sqlite3=SimpleNamespace(connect=connect,Row=sqlite3.Row)
        def canonical_path(sid,kind):
            require(Path(sid).name==sid,'Unsafe sample ID')
            return Path(scope['original_cache'])/hashlib.sha256(sid.encode()).hexdigest()[:2]/(sid+'.'+kind)
        pipeline.path_for=canonical_path
        frozen=pipeline.load_protocol
        def execution_protocol():
            value=dict(frozen());value.update(decode_threads_per_gpu=32,detector_services_per_gpu=32,inflight_groups_per_gpu=64,
                shared_memory=dict(value['shared_memory'],max_groups=64,aggregate_bound_gib=128))
            return value
        pipeline.load_protocol=execution_protocol
        runtime.enable_process_decode()
        from glare_lia_fused_fp32 import install
        install(pipeline,'fused2')
        pipeline.worker(7,None,False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--work',type=Path,required=True);p.add_argument('--plan',type=Path);p.add_argument('--export',type=Path)
    p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    if args.command=='prepare':require(args.plan is not None and args.export is not None,'prepare requires --plan and --export');prepare(args)
    else:run(args)


if __name__=='__main__':main()
