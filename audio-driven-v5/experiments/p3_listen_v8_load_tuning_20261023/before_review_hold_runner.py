"""Resumable full-source V8 segmentation/QC. Independent of frozen V7 outputs.

Bounded defaults: eight model workers, one per GPU, two preparation threads
per worker. Media rendering has a separate explicit concurrency limit.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import plan_listen_v8 as planner
import run_listen_v7_bulk as legacy
from p3_listen.common import read,write,sha,digest

ROOT=planner.ROOT
V7=legacy.OUT
OUT=Path(os.environ.get('LISTEN_V8_FULL_OUT',str(ROOT/'data_corpora/seamless_listen_v8_full')))
FEEDBACK=ROOT/'data_reviews/listener_v8_boundary_feedback.json'
SCIENCE=['scripts/listen_turns_v8.py','scripts/listen_policy_v8.py','scripts/plan_listen_v8.py']
KNOWN_SOURCE_FAILURES=('source_orientation_not_upright','orientation_ambiguous',
    'orientation_insufficient_landmarks','orientation_invalid_keypoint_shape',
    'Source AV/keypoint length disagreement','Missing transcript','source_missing_video_stream:')


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def verify():
    c=read(OUT/'contract.json')
    for name,h in c['code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen V8 dependency changed: '+name)
    for name,h in c['input_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Frozen V8 input changed: '+name)
    if sha(FEEDBACK)!=c['feedback_source_sha256']:raise ValueError('User feedback changed')
    if sha(ROOT/'data_reviews/listener_noise_rejections.json')!=c['legacy_feedback_sha256']:
        raise ValueError('Legacy feedback changed')
    return c


def init():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'contract.json').exists():verify();return
    legacy.verify_contract()
    accepted=read(ROOT/'experiments/p3_listen_v8_boundary_validation/reviewed_code_manifest.json')
    for name in SCIENCE:assert sha(ROOT/name)==accepted['files_sha256'][name],name
    for name in ('catalog.json','source_coverage.json','source_pair_index.json','source_exclusions.json'):
        shutil.copyfile(V7/name,OUT/name)
    catalog=read(OUT/'catalog.json')
    assert catalog['original_root']=='/public/0_DATA/2_Avatar/seamless_interaction'
    assert read(OUT/'source_coverage.json')['inventory_complete']
    assert sha(OUT/'source_pair_index.json')==catalog['source_index_sha256']
    feedback=read(FEEDBACK);assert feedback['boundary_policy_decision']=='accepted_for_current_workflow'
    rejects=read(V7/'human_rejections.json')['rows']
    rejects += [dict(r,decision='reject',original_review_decision=r['decision']) for r in feedback['rows']]
    write(OUT/'human_rejections.json',dict(rows=rejects))
    shutil.copyfile(V7/'human_retained.json',OUT/'human_retained.json')
    shutil.copyfile(FEEDBACK,OUT/'v8_boundary_feedback.json')
    files=sorted(set(legacy.CODE+SCIENCE+['scripts/run_listen_v8_full.py','scripts/render_listen_v8_full.py']))
    inputs=['catalog.json','source_pair_index.json','source_coverage.json','source_exclusions.json',
            'human_rejections.json','human_retained.json','v8_boundary_feedback.json']
    write(OUT/'contract.json',dict(revision='v8-full-1',utc=now(),video_policy=planner.POLICY,
        audio_policy=legacy.POLICY,normalization_policy=legacy.subturns.POLICY,
        code_sha256={f:sha(ROOT/f) for f in files},input_sha256={f:sha(OUT/f) for f in inputs},
        feedback_source_sha256=sha(FEEDBACK),legacy_feedback_sha256=sha(ROOT/'data_reviews/listener_noise_rejections.json'),
        models=read(V7/'contract.json')['models'],ast_batch=16,panns_batch=16,
        scope=catalog['scope'],training_ready=False,training_started=False,
        note='Accepted V8 segmentation; automatic boundary flags and audio QC remain separate.'))
    for name in ('logs','progress','results','errors','audit','normalized_pairs','telemetry','claims'):
        (OUT/name).mkdir(exist_ok=True)
    print(json.dumps(dict(initialized=str(OUT),pairs=len(catalog['rows']))),flush=True)


def configure_normalization():
    legacy.OUT=OUT
    legacy.subturns.OUT=OUT


def prepare(item):
    started=time.perf_counter()
    try:audit=planner.inspect_pair(item)
    except ValueError as e:
        if not str(e).startswith(KNOWN_SOURCE_FAILURES):raise
        audit=dict(pair=item['pair'],clips=[],unresolved=[dict(reason=str(e))],ok=False,
                   source_seconds=None,policy=planner.POLICY)
    audit['rejected']=audit['unresolved']
    for row in audit['clips']:
        row['listener_screen']=dict(accepted=True,deferred_reasons=
            ['listener_short_speech_unconfirmed'] if row['listener_compatibility']['uncertain_short_speech'] else [])
        row['duration_tier']='2to3s' if row['duration_s']<3 else 'over10s' if row['duration_s']>10 else 'ge3s'
    assert all(a['end']<=b['start']+1e-6 for a,b in zip(audit['clips'],audit['clips'][1:]))
    write(OUT/'audit'/(item['pair']+'.json'),audit)
    norm=None
    if audit['clips']:
        marker=OUT/'normalized_pairs'/(item['pair']+'.json')
        old=V7/'normalized_pairs'/(item['pair']+'.json')
        if not marker.exists() and old.exists():shutil.copyfile(old,marker)
        norm=legacy.normalize(item,audit['clips'])
    return audit,norm,time.perf_counter()-started


def screen(row,norm,ast,panns,rejects,positives):
    result=legacy.scan(row,norm,ast,panns,rejects,positives)
    result['audio_qc_status']=result['effective_decision']
    result['v8_combined_status']=('exclude' if result['effective_decision']=='exclude' or row['source_manual_rejections']
        else 'review' if result['effective_decision']=='review' or row['boundary_review_flags'] else 'candidate')
    result['boundary_policy_user_accepted']=True
    return result


def claim(queue,rank):
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(OUT/'queue_state.json');index=state['cursor']
        if index>=len(queue):return None
        pair=queue[index]
        write(OUT/'claims'/(pair+'.json'),dict(pair=pair,rank=rank,pid=os.getpid(),utc=now()))
        write(OUT/'queue_state.json',dict(cursor=index+1,total=len(queue)))
        return pair


def worker(rank,prep):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    contract=verify();configure_normalization()
    lookup={r['pair']:r for r in read(OUT/'catalog.json')['rows']}
    queue=read(OUT/'queue.json')['pairs']
    rejects=read(OUT/'human_rejections.json')['rows'];positives=read(OUT/'human_retained.json')['rows']
    counts=Counter();hours=Counter();timing=Counter();began=time.monotonic()
    def report(stage,**kw):
        value=dict(utc=now(),rank=rank,gpu=rank%8,pid=os.getpid(),stage=stage,
            prep_slots=prep,counts=dict(counts),audio_candidate_hours=dict(hours),
            timing_seconds=dict(timing),wall_seconds=time.monotonic()-began,**kw)
        write(OUT/'progress'/f'rank_{rank:03d}.json',value)
        print(json.dumps(value),flush=True)
    report('loading_models')
    ast=legacy.EventModel(legacy.MODEL,batch_size=contract['ast_batch'])
    panns=legacy.PannsModel(batch_size=contract['panns_batch'])
    errors=0
    with ThreadPoolExecutor(max_workers=prep) as pool:
        pending={}
        def fill():
            while len(pending)<prep:
                pair=claim(queue,rank)
                if pair is None:break
                pending[pool.submit(prepare,lookup[pair])]=pair;counts['claimed_pairs']+=1
        fill()
        while pending:
            started=time.perf_counter();done,_=wait(pending,return_when=FIRST_COMPLETED)
            timing['waiting_for_preparation']+=time.perf_counter()-started
            for future in done:
                pair=pending.pop(future);item=lookup[pair];fill()
                report('screening',pair=pair,inflight=len(pending))
                try:
                    audit,norm,elapsed=future.result();timing['preparation_sum']+=elapsed
                    started=time.perf_counter()
                    clips=[screen(r,norm,ast,panns,rejects,positives) for r in audit['clips']]
                    timing['audio_screening']+=time.perf_counter()-started
                    write(OUT/'results'/(pair+'.json'),dict(pair=pair,split=item['split'],group_id=item['group_id'],
                        complete=True,utc=now(),clips=clips,rejected=audit['unresolved'],source_ok=audit['ok'],
                        source_seconds=audit.get('source_seconds'),training_ready=False))
                    for row in clips:
                        key=item['split']+'/'+row['effective_decision'];counts[key]+=1;hours[key]+=row['duration_s']/3600
                    counts['complete_pairs']+=1;errors=0
                except Exception as e:
                    errors+=1;counts['error_pairs']+=1
                    write(OUT/'errors'/(pair+'.json'),dict(pair=pair,utc=now(),error=repr(e),traceback=traceback.format_exc()))
                    if errors>=10:raise RuntimeError('Ten consecutive errors; inspect logs') from e
                report('pair_finished',pair=pair)
    report('finished')


def cpu_sample():
    allowed=os.sched_getaffinity(0);total=idle=io=0
    for line in Path('/proc/stat').read_text().splitlines():
        key,*v=line.split()
        if key.startswith('cpu') and key[3:].isdigit() and int(key[3:]) in allowed:
            v=list(map(int,v));total+=sum(v[:8]);idle+=v[3]+v[4];io+=v[4]
    return total,idle,io


def telemetry(previous):
    cur=cpu_sample();delta=cur[0]-previous[0]
    p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
    gpus=[dict(zip(('gpu','utilization_percent','memory_mib','total_mib'),map(int,s.split(','))))
          for s in p.stdout.strip().splitlines()] if p.returncode==0 else []
    states=[read(p) for p in (OUT/'progress').glob('rank_*.json')]
    value=dict(utc=now(),unix=time.time(),gpus=gpus,
        cpu_busy_percent=100*(1-(cur[1]-previous[1])/max(1,delta)),
        cpu_iowait_percent=100*(cur[2]-previous[2])/max(1,delta),
        complete_pairs=sum(s['counts'].get('complete_pairs',0) for s in states),
        error_pairs=sum(s['counts'].get('error_pairs',0) for s in states),
        active_workers=sum(s['stage']!='finished' for s in states),
        preparation_slots=sum(s['prep_slots'] for s in states if s['stage']!='finished'))
    write(OUT/'telemetry/latest.json',value)
    with (OUT/'telemetry/history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    return cur


def report(export=False):
    expected={r['pair'] for r in read(OUT/'catalog.json')['rows']}
    counts=Counter();hours=Counter();combined=Counter();flags=Counter();reasons=Counter();done=set();durations=Counter()
    handles={}
    if export:(OUT/'manifests').mkdir(exist_ok=True)
    try:
        for path in (OUT/'results').glob('*.json'):
            r=read(path)
            if not r.get('complete'):continue
            assert r['pair'] in expected
            done.add(r['pair'])
            for row in r['clips']:
                key='_'.join((row['split'],row['effective_decision'],row['admission_pool']))
                counts[key]+=1;hours[key]+=row['duration_s']/3600
                combined[row['split']+'/'+row['v8_combined_status']]+=row['duration_s']/3600
                flags.update(row['boundary_review_flags'])
                durations['2-3s' if row['duration_s']<3 else '3-10s' if row['duration_s']<=10 else '10-11.95s']+=1
                if export:
                    if key not in handles:handles[key]=(OUT/'manifests'/(key+'.jsonl.pending')).open('w')
                    compact={k:v for k,v in row.items() if k not in ('qc3','normalized_audio','human_decisions')}
                    compact['normalized_audio']={role:{k:t[k] for k in ('path','sha256','gain_db')} for role,t in row['normalized_audio'].items()}
                    compact['quality_evidence_file']=str(path)
                    handles[key].write(json.dumps(compact,ensure_ascii=False)+'\n')
            for bad in r['rejected']:reasons[str(bad.get('reason','unknown'))]+=1
    finally:
        for h in handles.values():h.close()
    for key in handles:(OUT/'manifests'/(key+'.jsonl.pending')).replace(OUT/'manifests'/(key+'.jsonl'))
    errors={p.stem for p in (OUT/'errors').glob('*.json')}-done
    value=dict(utc=now(),total_pairs=len(expected),complete_pairs=len(done),error_pairs=len(errors),
        pending_pairs=len(expected-done-errors),counts=dict(counts),audio_candidate_hours=dict(hours),
        combined_status_hours=dict(combined),boundary_flag_counts=dict(flags),duration_counts=dict(durations),
        unresolved_reason_counts=dict(reasons),training_ready=False,
        note='Audio keep is not blanket training admission. Boundary flags remain. Unknown/error sources are not classified as noise.')
    write(OUT/'screening_summary.json',value)
    if export:write(OUT/'manifests/index.json',value)
    return value


def supervise(workers,prep,limit=0,render=True,render_workers=16):
    verify()
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=read(OUT/'catalog.json')['rows']
        done={p.stem for p in (OUT/'results').glob('*.json')}
        queue=[r['pair'] for r in rows if r['pair'] not in done]
        if limit:queue=queue[:limit]
        write(OUT/'queue.json',dict(pairs=queue));write(OUT/'queue_state.json',dict(cursor=0,total=len(queue)))
        # Recover orphan claims after supervisor/worker failures by rebuilding from actual results.
        for p in (OUT/'progress').glob('rank_*.json'):p.unlink()
        children=[]
        for rank in range(workers):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank%8),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            with (OUT/'logs'/f'rank_{rank:03d}.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--rank',str(rank),'--prep',str(prep)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            children.append(p)
        write(OUT/'runtime.json',dict(utc=now(),state='screening',supervisor_pid=os.getpid(),workers=workers,
            prep_per_worker=prep,total_preparation_slots=workers*prep,gpus=8,pids=[p.pid for p in children],
            queue_pairs=len(queue),render_after_screening=render,render_workers=render_workers))
        previous=cpu_sample();last_report=0
        while any(p.poll() is None for p in children):
            time.sleep(15)
            previous=telemetry(previous)
            if time.monotonic()-last_report>120:report();last_report=time.monotonic()
        codes=[p.wait() for p in children];summary=report(export=True)
        write(OUT/'screening_exit.json',dict(utc=now(),exit_codes=codes,summary=summary))
        if any(codes):
            write(OUT/'runtime.json',dict(state='screening_worker_failure',utc=now(),exit_codes=codes))
            raise RuntimeError('Workers failed; resume after inspecting errors. Completed pairs are retained.')
        if render and not limit:
            write(OUT/'runtime.json',dict(state='rendering',utc=now(),supervisor_pid=os.getpid(),workers=render_workers))
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/render_listen_v8_full.py'),'--workers',str(render_workers)],check=True)
        write(OUT/'runtime.json',dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',utc=now(),
            summary=summary,render_requested=render and not limit,training_started=False))


def detach(args):
    verify()
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'run','--workers',str(args.workers),
        '--prep',str(args.prep),'--render-workers',str(args.render_workers)]
    if args.limit:command+=['--limit',str(args.limit)]
    if args.no_render:command+=['--no-render']
    with (OUT/'logs/supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(utc=now(),pid=p.pid,command=command));print(json.dumps(read(OUT/'launch.json')),flush=True)


def argument_parser():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','run','detach','worker','report'])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--prep',type=int,default=2)
    p.add_argument('--render-workers',type=int,default=16)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int,default=0);p.add_argument('--no-render',action='store_true')
    return p


if __name__=='__main__':
    p=argument_parser()
    a=p.parse_args()
    if a.workers<8 or a.workers%8 or a.prep<1:p.error('workers must be a positive multiple of 8; prep >=1')
    if a.render_workers<1:p.error('render-workers must be positive')
    if a.stage=='init':init()
    elif a.stage=='worker':worker(a.rank,a.prep)
    elif a.stage=='run':supervise(a.workers,a.prep,a.limit,not a.no_render,a.render_workers)
    elif a.stage=='detach':detach(a)
    else:print(json.dumps(report(export=True)),flush=True)
