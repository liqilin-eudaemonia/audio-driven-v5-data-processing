"""Two-node quality-first pipeline; old J inputs and annotation results stay intact."""
import copy
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from listen_benchmark.j_annotation import JobQueue, acquire, read, write, result_path as label_path
from listen_visual_quality.io import canonical_hash, file_hash
from listen_visual_quality.cli import load_policy, DEFAULT_MODEL, check_source
from listen_visual_quality.decision import evidence_index, decide

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/listen_j_quality_pipeline.py'
QUALITY_PYTHON = '/tmp/listen-j-quality-env/bin/python'
LABEL_PYTHON = '/private/software/conda/envs/vllm/bin/python'


def live(record):
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_ticks'])
    except (FileNotFoundError, KeyError):
        return False


def identity(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def compact_write(path, data):
    """Atomic compact JSON avoids pretty-print CPU and large measurement output."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def measurement_path(work, sid):
    return Path(work) / 'measurements' / hashlib.sha256(sid.encode()).hexdigest()[:2] / (sid + '.json')


def cropped_row(row, record, decision, receipt_path):
    if decision['decision'] != 'retain' or record['status'] != 'ok':
        raise ValueError('Only measured retain rows may enter annotation')
    if record['source'] != row['source'] or decision['source_sha256'] != canonical_hash(row['source']):
        raise ValueError('Quality/source mismatch')
    new = copy.deepcopy(row)
    old = row['source']['video_transform']['crop']
    crop = record['crop']['new']; width, height = row['source']['video_transform']['source_size']
    if (any(type(crop[k]) is not int for k in ('x','y','width','height')) or
        crop['width'] != crop['height'] or crop['width'] <= 0 or
        min(crop['x'], crop['y']) < 0 or crop['x']+crop['width'] > width or
        crop['y']+crop['height'] > height or crop['width'] > old['width']*1.05+1e-9):
        raise ValueError('Invalid bounded crop')
    new['source']['video_transform']['crop'] = copy.deepcopy(crop)
    new['source']['crop_proposal'] = dict(x=crop['x'], y=crop['y'], size=crop['width'],
        status='proposal_only', visual_filter_applied=True,revision='quality-bounded-crop-v1-20261030')
    new['source']['quality_screening'] = {
        'measurement': str(receipt_path), 'input_sha256': decision['source_sha256'],
        'decision': 'retain', 'crop_applied': crop != old,
        'original_crop': old, 'original_proposal':row['source']['crop_proposal'],
        'policy': 'listen-visual-quality-v1-20261030',
        'training_admitted': False,
    }
    return new


def runtime_contract(work):
    files = sorted((ROOT/'src/listen_visual_quality').glob('*.py')) + [Path(__file__), ENTRY,
        Path(__file__).with_name('quality_decode_session.py'),Path(__file__).with_name('quality_cpu_execution.py'),
        ROOT/'configs/listen_visual_quality_v1.json',
        ROOT/'configs/listen_visual_quality_reviewed_evidence_20261030.json',
        ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py',
        DEFAULT_MODEL]
    data = {'code': {str(p.relative_to(ROOT)): file_hash(p) for p in files},
            'scope_sha256': file_hash(Path(work)/'ready.jsonl'),
            'packages': {n: importlib.metadata.version(n) for n in ('av','numpy','mediapipe','opencv-contrib-python')},
            'apply_bounded_crop': True, 'reuse_existing_annotations': True, 'backend': 'cpu',
            'sequential_decoder_reuse':True,'detector_graph_threads':1}
    return {'sha256': canonical_hash(data), 'contract': data}


def source_job(work, config, job_id, policy, evidence, fingerprint, progress=None):
    from listen_benchmark.quality_cpu_execution import bounded_detector_threads
    from listen_benchmark.quality_decode_session import sequential_measurements
    bounded=config.get('bounded_detector_threads',False)
    sequential=config.get('sequential_decode',False)
    with bounded_detector_threads(str(ROOT)) if bounded else nullcontext():
        with sequential_measurements(str(ROOT)) if sequential else nullcontext() as stats:
            receipt=_source_job(work,config,job_id,policy,evidence,fingerprint,progress)
            receipt['decoder_stats']=dict(stats) if stats is not None else None
            return receipt


def _source_job(work, config, job_id, policy, evidence, fingerprint, progress=None):
    old = Path(config['old_work'])
    job_path = old/'jobs'/f'{job_id}.json'
    job = read(job_path)
    checked = {}; approved = []; decisions = []; reuse = {'vlm':0,'nod':0}
    from listen_visual_quality.measurement import measure_row
    ordered=sorted(job['rows'],key=lambda r:(r['source']['listener_id'],r['source']['source_start_s']))
    for row in ordered:
        sid = row['sample_id']; check_source(row, checked)
        path = measurement_path(work, sid)
        record = read(path) if path.exists() else None
        if record and (record.get('measurement_hash') != fingerprint or record['source'] != row['source']):
            raise ValueError('Incompatible measurement cache: '+sid)
        cached = bool(record and record['status']=='ok')
        if not cached:
            record = measure_row(row, policy, str(ROOT), str(DEFAULT_MODEL))
            record['measurement_hash'] = fingerprint
            compact_write(path, record)
        if record['status'] != 'ok':
            # Retry the source via the recoverable queue. Completed sibling clips are cached.
            raise RuntimeError(f"{sid}: {record.get('error')}")
        decision = decide(record, policy, evidence)
        decision['measurement'] = str(path)
        decision['annotation_reuse'] = {}
        if decision['decision'] == 'retain':
            row_new = cropped_row(row, record, decision, path)
            for branch in reuse:
                previous = label_path(old, branch, sid)
                if previous.exists():
                    label = read(previous)
                    # Crop changes are explicitly permitted, but times and participant must match.
                    original = label.get('source', {})
                    for key in ('sample_id','source_start_s','source_end_s','source','listener_id'):
                        if original.get(key) != row['source'].get(key):
                            raise ValueError('Existing annotation source mismatch: '+sid)
                    decision['annotation_reuse'][branch] = {
                        'path': str(previous), 'sha256': file_hash(previous),
                        'crop_matches_final': original.get('video_transform') == row_new['source']['video_transform'],
                        'original_transform': original.get('video_transform'),
                        'user_authorized_no_reannotation': True}
                    reuse[branch] += 1
            approved.append(row_new)
        decisions.append(decision)
        if progress: progress(sid, cached, decision['decision'])
    counts = {k:sum(d['decision']==k for d in decisions) for k in ('retain','exclude','review')}
    output = dict(job, rows=approved, visual_quality_applied=True,
                  original_job=str(job_path), original_job_sha256=file_hash(job_path),
                  quality_decisions=decisions, quality_counts=counts, existing_annotation_reuse=reuse)
    compact_write(Path(work)/'approved_jobs'/f'{job_id}.json', output)
    return {'job_id':job_id,'clips':len(decisions),'counts':counts,'annotation_reuse':reuse,
            'approved_job_sha256':file_hash(Path(work)/'approved_jobs'/f'{job_id}.json'),
            'contract_sha256':fingerprint,'at':time.time()}


def worker(work, slot):
    work = Path(work); cfg = read(work/'pipeline_config.json')
    cores=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cores[slot % len(cores)]})
    owner = acquire(work/'workers'/f'{slot}.lock', False)
    if owner is None:return
    policy = load_policy(ROOT/'configs/listen_visual_quality_v1.json')
    evidence = evidence_index(read(ROOT/'configs/listen_visual_quality_reviewed_evidence_20261030.json'))
    contract = read(work/'contract.json')['sha256']
    queue = JobQueue(work, 'quality'); stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    state = {'pid':os.getpid(),'start_ticks':identity(os.getpid()),'slot':slot,
             'completed_clips':0,'reused_clips':0,'started_at':time.time()}
    def progress(sid, cached, decision):
        state.update(sample_id=sid, decision=decision, heartbeat=time.time())
        state['completed_clips']+=1;state['reused_clips']+=int(cached)
        compact_write(work/'workers'/f'{slot}.json', state)
    try:
        while not stop:
            claim = queue.claim(f'quality-{slot}')
            if claim is None:break
            job, handle = claim
            try:
                receipt = source_job(work,cfg,job,policy,evidence,contract,progress)
                queue.finish(job,receipt=receipt)
            except Exception as error:
                compact_write(work/'errors'/f'{job}.json', {'error':repr(error),'at':time.time()})
                queue.finish(job,error=repr(error))
            finally:handle.close()
    finally:
        state['ended_at']=time.time();compact_write(work/'workers'/f'{slot}.json',state);owner.close()


def export_node(work, cfg):
    """Build a fresh annotation queue; old result stores are reused without copying or overwriting."""
    work=Path(work); target=Path(cfg['annotation_work']);old=Path(cfg['old_work'])
    target.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','selected.json','resource_tuning.json'):
        shutil.copy2(work/('original_'+name),target/name)
    config=read(target/'config.json')
    config['expected_hostname']=cfg['hostname']
    config.update(quality_input=str(work),previous_annotation_work=str(old),
                  reuse_completed_annotations=True,quality_first=True)
    config['warm_cache']='/dev/shm/listen_j_quality_annotation_'+cfg['node']+'_20261030'
    write(target/'config.json',config)
    (target/'jobs').mkdir(exist_ok=True)
    for branch in ('vlm','nod'):
        (target/branch).mkdir(exist_ok=True)
        link=target/branch/'results'
        if not link.exists():link.symlink_to(old/branch/'results',target_is_directory=True)
        if link.resolve() != (old/branch/'results').resolve():raise ValueError('Incorrect annotation result store')
    counts={'retain':0,'exclude':0,'review':0}; reuse={'vlm':0,'nod':0}; jobs=0
    failed=read(work/'queues/quality/state.json')['failed']
    outputs={n:(work/(n+'.tmp')).open('w') for n in ('decisions.jsonl','retained_rows.jsonl','excluded_rows.jsonl','review_rows.jsonl')}
    ready_tmp=target/'ready.jsonl.tmp'
    try:
        with ready_tmp.open('w') as ready:
            for line in (work/'ready.jsonl').read_text().splitlines():
                jid=json.loads(line)['job_id']
                if jid in failed:
                    for row in read(old/'jobs'/f'{jid}.json')['rows']:
                        d={'sample_id':row['sample_id'],'decision':'review','reasons':['measurement_execution_error_retry_required'],
                           'source_sha256':canonical_hash(row['source']),'source_failure':failed[jid]}
                        counts['review']+=1
                        outputs['decisions.jsonl'].write(json.dumps(d)+'\n')
                        outputs['review_rows.jsonl'].write(json.dumps({'row':row,'decision':d})+'\n')
                    continue
                p=work/'approved_jobs'/f'{jid}.json';job=read(p)
                receipt=read(work/'quality/completed'/f'{jid}.json')
                if receipt['approved_job_sha256'] != file_hash(p):raise ValueError('Uncommitted screened job')
                for k in counts:counts[k]+=job['quality_counts'][k]
                for k in reuse:reuse[k]+=job['existing_annotation_reuse'][k]
                original={r['sample_id']:r for r in read(old/'jobs'/f'{jid}.json')['rows']}
                for d in job['quality_decisions']:
                    outputs['decisions.jsonl'].write(json.dumps(d)+'\n')
                    if d['decision']!='retain':
                        name='excluded_rows.jsonl' if d['decision']=='exclude' else 'review_rows.jsonl'
                        outputs[name].write(json.dumps({'row':original[d['sample_id']],'decision':d})+'\n')
                for row in job['rows']:outputs['retained_rows.jsonl'].write(json.dumps(row)+'\n')
                if job['rows']:
                    annotation_job={k:v for k,v in job.items() if k!='quality_decisions'}
                    compact_write(target/'jobs'/f'{jid}.json',annotation_job)
                    ready.write(json.dumps({'job_id':jid,'count':len(job['rows'])})+'\n');jobs+=1
    finally:
        for f in outputs.values():f.close()
    for name in outputs:(work/(name+'.tmp')).replace(work/name)
    ready_tmp.replace(target/'ready.jsonl')
    summary={'at':time.time(),'processed':sum(counts.values()),'decisions':counts,'reused_annotation_branches':reuse,
             'annotation_jobs':jobs,'failed_sources':len(failed),'annotation_work':str(target),
             'crop_proposals_applied':True,'training_admitted':False}
    if summary['processed']!=cfg['scope_clips']:raise ValueError('Screened scope count mismatch')
    write(target/'ingest_complete.json',{'ready_jobs':jobs,'ready_clips':counts['retain'],'quality_input':str(work)})
    write(target/'prime_complete.json',{'skipped':True,'reason':'Existing accelerated worker handles prefetch'})
    if not (target/'supervisor.json').exists():
        write(target/'supervisor.json',{'phase':'annotation','started_at':time.time(),
            'coverage_deadline':time.time()+8*3600,'children':{},'events':[],'benchmark_attempt':0})
    write(work/'quality_complete.json',summary)
    return summary


def spawn(work, name, command):
    path=Path(work)/'logs'/f'{name}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,
            CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    return {'pid':child.pid,'start_ticks':identity(child.pid),'at':time.time(),'command':command}


def cpu_usage():
    return int(dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])


def controller(work):
    work=Path(work);cfg=read(work/'pipeline_config.json')
    if socket.gethostname()!=cfg['hostname']:raise ValueError('Wrong node')
    owner=acquire(work/'controller.lock',False)
    if owner is None:return
    state=read(work/'controller.json') if (work/'controller.json').exists() else {'children':{},'started_at':time.time()}
    state.update(pid=os.getpid(),start_ticks=identity(os.getpid()))
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping)
    last_time=time.time();last_usage=cpu_usage();last_completed=0
    queue=JobQueue(work,'quality')
    try:
        while not stop:
            now=time.time();state['heartbeat']=now
            # Never compete with legacy annotation or switch its result store while workers are alive.
            previous=read(work/'pause_receipt.json')['children']
            draining=[n for n,c in previous.items() if n.startswith(('gpu-','nod-')) and live(c)]
            if draining:
                state.update(phase='draining_previous_annotation',draining=draining)
            elif not (work/'quality_complete.json').exists():
                state['phase']='cpu_quality'
                q=read(queue.path);completed=q['completed'];failed=len(q['failed'])
                remaining=cfg['scope_sources']-completed-failed
                usage=cpu_usage();dt=now-last_time
                state.update(completed_sources=completed,completed_clips=q.get('completed_clips',0),
                    failed_sources=failed,remaining_sources=remaining,
                    cpu_quota_utilization=(usage-last_usage)/1e6/max(dt,1e-6)/cfg['cpu_quota'],
                    clips_per_s=(q.get('completed_clips',0)-last_completed)/max(dt,1e-6))
                last_time=now;last_usage=usage;last_completed=q.get('completed_clips',0)
                workers_cfg=read(work/'worker_control.json')
                target=workers_cfg['workers'];state['target_workers']=target
                active=sum(live(c) for c in state['children'].values())
                if remaining==0 and active==0:
                    state['phase']='exporting_filtered_annotation_queue';write(work/'controller.json',state)
                    export_node(work,cfg)
                else:
                    # Rapid bounded startup; one worker per assigned CPU until the target is reached.
                    budget=32
                    for slot in range(target):
                        key=str(slot)
                        if not live(state['children'].get(key,{})) and active<min(target,remaining) and budget:
                            state['children'][key]=spawn(work,f'quality-{slot}',[QUALITY_PYTHON,str(ENTRY),'worker',
                                '--work',str(work),'--slot',str(slot)])
                            budget-=1;active+=1
                    state['active_workers']=active
            else:
                nodes=read(work.parent/'nodes.json')['nodes']
                waiting=[n for n,item in nodes.items() if not (Path(item['quality_work'])/'quality_complete.json').exists()]
                if waiting:
                    state.update(phase='helping_other_node_quality',waiting=waiting)
                    # Filesystem source locks keep ownership unique. Helpers write to the
                    # original quality shard, preserving annotation/source partition identity.
                    other=Path(nodes[waiting[0]]['quality_work'])
                    target=read(work/'worker_control.json')['workers'];budget=32
                    for slot in range(target):
                        key='helper-'+str(slot)
                        if not live(state['children'].get(key,{})) and budget:
                            state['children'][key]=spawn(work,key,[QUALITY_PYTHON,str(ENTRY),'worker',
                                '--work',str(other),'--slot',str(1000+slot)])
                            budget-=1
                else:
                    state['phase']='annotation_resumed'
                    target=Path(cfg['annotation_work'])
                    watch=read(target/'watchdog.json') if (target/'watchdog.json').exists() else {}
                    if not live(watch):
                        state['annotation_watchdog']=spawn(work,'annotation-watchdog',[LABEL_PYTHON,
                            str(ROOT/'scripts/listen_j_annotation_watchdog.py'),'--work',str(target),
                            '--upstream',str(ROOT/'data_corpora/seamless_listen_j_full_20261029')])
                    write(work/'controller.json',state)
                    break
            compact_write(work/'controller.json',state)
            with (work/'telemetry.jsonl').open('a') as f:
                f.write(json.dumps({k:v for k,v in state.items() if k!='children'})+'\n')
            time.sleep(10)
    finally:
        state['controller_ended_at']=time.time();write(work/'controller.json',state);owner.close()


def watchdog(work):
    work=Path(work);owner=acquire(work/'watchdog.lock',False)
    if owner is None:return
    state={'pid':os.getpid(),'start_ticks':identity(os.getpid()),'started_at':time.time()}
    while not (work/'STOP').exists():
        c=read(work/'controller.json') if (work/'controller.json').exists() else {}
        if c.get('phase')=='annotation_resumed':break
        if not live(c):
            state['controller']=spawn(work,'controller',[QUALITY_PYTHON,str(ENTRY),'controller','--work',str(work)])
        state['heartbeat']=time.time();write(work/'watchdog.json',state);time.sleep(15)
    state['ended_at']=time.time();write(work/'watchdog.json',state);owner.close()
