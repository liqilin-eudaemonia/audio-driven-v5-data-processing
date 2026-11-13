"""Reviewed QC4 full run, isolated from earlier V8 results and review artifacts.

Eight GPUs, one model worker and one preparation slot each. Only candidate
media are rendered (four tasks); repair/content/role pools remain references.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_full as full
import listen_audio_quality_v4 as qc4
import listen_v8_release as release
from p3_listen.common import read,write,sha

ROOT=full.ROOT
PARENT=ROOT/'data_corpora/seamless_listen_v8_full'
SAMPLE=ROOT/'data_corpora/seamless_listen_v8_qc4_sample'
OUT=Path(os.environ.get('LISTEN_V8_QC4_OUT',str(ROOT/'data_corpora/seamless_listen_v8_qc4_full')))
FEEDBACK=ROOT/'data_reviews/listener_v8_qc4_feedback.json'
SCRIPT=Path(__file__).resolve()
BASE_SCREEN=full.screen

def assess(row,feedback):
    if 'qc4' not in row:
        waves={role:full.legacy.read_wave(row['normalized_audio'][role]['path'],row['start'],row['duration_s']) for role in ('speaker','listener')}
        row['qc4']=qc4.assess(row,waves['speaker'],waves['listener'])
    row['release']=release.route(row,feedback)
    return row

def install():
    full.OUT=OUT
    # Base supervisor launches this wrapper, so every child installs QC4/routing.
    full.__file__=str(SCRIPT)
    feedback=read(OUT/'qc4_review_feedback.json')['rows']
    def screen(*args,**kwargs):return assess(BASE_SCREEN(*args,**kwargs),feedback)
    full.screen=screen
    full.report=report

def init():
    if (OUT/'contract.json').exists():
        install();full.verify();return
    full.OUT=PARENT;parent=full.verify()
    full.OUT=SAMPLE;full.verify()
    feedback=read(FEEDBACK)
    assert feedback['full_run_authorized'] and feedback['quality_policy_accepted']
    assert sha(feedback['sample_manifest'])==feedback['sample_manifest_sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ('results','errors','audit','normalized_pairs','logs','progress','telemetry','claims','manifests','render_results','render_errors'):(OUT/d).mkdir(exist_ok=True)
    for name in parent['input_sha256']:shutil.copyfile(PARENT/name,OUT/name)
    shutil.copyfile(FEEDBACK,OUT/'qc4_review_feedback.json')
    contract=dict(parent,revision='v8-qc4-full-reviewed-1',utc=full.now(),
        parent_contract_sha256=sha(PARENT/'contract.json'),sample_contract_sha256=sha(SAMPLE/'contract.json'),
        additional_policy=qc4.POLICY,release_policy_revision=release.REVISION,
        code_sha256=dict(parent['code_sha256']),input_sha256=dict(parent['input_sha256']),
        workers=8,prep_per_worker=1,render_workers=4,render_pools=['candidate'],
        training_ready=False,repair_reserve_auto_processing=False)
    for f in ('scripts/listen_audio_quality_v4.py','scripts/listen_v8_release.py','scripts/run_listen_v8_qc4_full.py'):contract['code_sha256'][f]=sha(ROOT/f)
    contract['input_sha256']['qc4_review_feedback.json']=sha(OUT/'qc4_review_feedback.json')
    write(OUT/'contract.json',contract)
    write(OUT/'review_hold.json',dict(state='accepted_with_quarantines',full_run_authorized=True,utc=full.now(),feedback_sha256=sha(FEEDBACK)))
    # Reuse completed independent batch; migrate the five known problem pairs
    # with the identical QC4 algorithm, without overwriting their original data.
    source_paths=list(sorted((SAMPLE/'results').glob('*.json')))
    known_pairs={r['pair'] for r in feedback['rows'] if r['page_number']>=19}
    source_paths += [PARENT/'results'/(pair+'.json') for pair in sorted(known_pairs)]
    migrated={}
    for src in source_paths:
        r=read(src)
        for row in r['clips']:assess(row,feedback['rows'])
        r['qc4_migration']=dict(source=str(src),sha256=sha(src),utc=full.now())
        write(OUT/'results'/src.name,r);migrated[r['pair']]=r['qc4_migration']
        for folder in ('audit','normalized_pairs'):
            cached=src.parent.parent/folder/src.name
            if cached.exists():shutil.copyfile(cached,OUT/folder/src.name)
    write(OUT/'migration.json',dict(pairs=migrated,source_pairs=len(source_paths)))
    install();full.verify();report(export=True)
    write(OUT/'runtime.json',dict(state='ready_for_authorized_full_run',utc=full.now(),workers=0,seed_pairs=len(migrated),training_started=False))

def report(export=False):
    expected={r['pair'] for r in read(OUT/'catalog.json')['rows']}
    done=set();counts=Counter();hours=Counter();source_reasons=Counter();handles={}
    try:
        for path in (OUT/'results').glob('*.json'):
            result=read(path)
            if not result.get('complete'):continue
            assert result['pair'] in expected
            done.add(result['pair'])
            for row in result['clips']:
                assert row['release']['revision']==release.REVISION
                pool=row['release']['pool'];key='_'.join((row['split'],pool,row['admission_pool']))
                counts[key]+=1;hours[key]+=row['duration_s']/3600
                if export:
                    if key not in handles:handles[key]=(OUT/'manifests'/(key+'.jsonl.pending')).open('w')
                    compact={k:v for k,v in row.items() if k not in ('qc3','qc4')}
                    compact.update(quality_evidence_file=str(path),qc4_status=row['qc4']['status'])
                    handles[key].write(json.dumps(compact,ensure_ascii=False)+'\n')
            source_reasons.update(str(r.get('reason','unknown')) for r in result['rejected'])
    finally:
        for h in handles.values():h.close()
    if export:
        for key in handles:(OUT/'manifests'/(key+'.jsonl.pending')).replace(OUT/'manifests'/(key+'.jsonl'))
        # Remove only stale derived manifests, never raw results or media.
        for old in (OUT/'manifests').glob('*.jsonl'):
            if old.stem not in handles and not old.stem.startswith('rendered_'):old.unlink()
    errors={p.stem for p in (OUT/'errors').glob('*.json')}-done
    value=dict(utc=full.now(),total_pairs=len(expected),complete_pairs=len(done),error_pairs=len(errors),pending_pairs=len(expected-done-errors),
        pool_counts=dict(counts),pool_hours=dict(hours),unresolved_reason_counts=dict(source_reasons),
        candidate_is_automatic_unless_exact_human_approval=True,training_ready=False,
        note='Repair reserve is quarantined; no automatic denoising or release. Primary and onset reserve are counted separately.')
    write(OUT/'screening_summary.json',value)
    if export:write(OUT/'manifests/index.json',value)
    return value

def render():
    import render_listen_v8_full as renderer
    renderer.OUT=OUT
    full.verify()
    with (OUT/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        counts=Counter();failures=[]
        def pair(path):
            result=read(path);items=[];verified=set()
            for row in result['clips']:
                if row['release']['pool']!='candidate':continue
                # Export path reflects release pool; retain original QC3 fields.
                item=renderer.export(dict(row,effective_decision='candidate'),verified)
                item['release']=row['release'];items.append(item)
            write(OUT/'render_results'/path.name,dict(pair=result['pair'],clips=items,complete=True))
            return items
        paths=sorted((OUT/'results').glob('*.json'))
        def attempt(path):
            try:return path,pair(path),None
            except Exception as e:return path,[],repr(e)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for path,items,error in pool.map(attempt,paths):
                if error:
                    failures.append(path.stem);write(OUT/'render_errors'/path.name,dict(error=error,pair=path.stem))
                else:
                    counts['complete_pairs']+=1;counts['candidate_clips']+=len(items)
                write(OUT/'render_summary.json',dict(utc=full.now(),counts=dict(counts),error_pairs=len(failures),workers=4,pools=['candidate'],training_ready=False))
        handles={}
        try:
            for path in (OUT/'render_results').glob('*.json'):
                for row in read(path)['clips']:
                    key='rendered_'+row['split']+'_candidate_'+row['admission_pool']
                    if key not in handles:handles[key]=(OUT/'manifests'/(key+'.jsonl.pending')).open('w')
                    handles[key].write(json.dumps(row,ensure_ascii=False)+'\n')
        finally:
            for h in handles.values():h.close()
        for key in handles:(OUT/'manifests'/(key+'.jsonl.pending')).replace(OUT/'manifests'/(key+'.jsonl'))
        if failures:raise RuntimeError('Candidate render errors; inspect render_errors')

def run(no_render=False):
    install();full.verify()
    if shutil.disk_usage(OUT).free<200*1024**3:raise RuntimeError('Less than 200 GiB free; inspect disk before processing')
    full.supervise(8,1,render=False,render_workers=4)
    if not no_render:
        write(OUT/'runtime.json',dict(state='rendering_candidates',utc=full.now(),workers=4,supervisor_pid=os.getpid()))
        render()
    summary=report(export=True)
    write(OUT/'runtime.json',dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',utc=full.now(),workers=0,summary=summary,training_started=False))

def detach(no_render=False):
    install();full.verify();full.check_review_hold(None)
    command=[sys.executable,'-u',str(SCRIPT),'run']+(['--no-render'] if no_render else [])
    with (OUT/'logs/supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write(OUT/'launch.json',dict(utc=full.now(),pid=p.pid,command=command));print(json.dumps(read(OUT/'launch.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','worker','run','detach','report','render'])
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--prep',type=int,default=1)
    parser.add_argument('--no-render',action='store_true');args=parser.parse_args()
    if args.stage=='init':init()
    elif args.stage=='detach':detach(args.no_render)
    elif args.stage=='run':run(args.no_render)
    elif args.stage=='worker':
        assert 0<=args.rank<8 and args.prep==1
        install();full.worker(args.rank,1)
    elif args.stage=='render':install();render()
    else:install();print(json.dumps(report(export=True)),flush=True)
