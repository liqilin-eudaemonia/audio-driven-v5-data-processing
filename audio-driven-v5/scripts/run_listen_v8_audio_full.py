"""Authorized full audio-first run, with native face statistics and no rendering.

Frozen science modules are reused without editing their files. This adapter
owns a separate queue, contract, results, and resumable per-pair summaries.
"""
import argparse
import collections
import concurrent.futures
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import run_listen_v8_cluster as cluster
import run_listen_v8_qc4_full as original
import plan_listen_v8_audio_full as planner
import listen_v8_audio_final as policy
import listen_v8_recovery as recovery
import listen_v8_face_statistics as faces
from p3_listen.common import read, write, sha

full = original.full
ROOT = full.ROOT
BASE = ROOT/'data_corpora/seamless_listen_v8_qc4_full'
RECOVERY = ROOT/'experiments/p3_listen_v8_recovery_20261023'
OUT = ROOT/'data_corpora/seamless_listen_v8_audio_final_full'
SCRIPT = Path(__file__).resolve()
SCHED = OUT/'cluster'
REVISION = 'v8-audio-final-full-no-face-filter-1'
CONFIG = ROOT/'configs/p3_listen_v8_cluster.json'
FEEDBACK = None
OLD_FEEDBACK = None
RAW_SCREEN = original.BASE_SCREEN
ORIGINAL_VERIFY = full.verify
KNOWN_FAILURES = ('source_missing_video_stream:', 'Source AV length disagreement', 'Missing transcript')


def verify():
    contract = read(OUT/'contract.json')
    for path, expected in contract['code_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen full-run code changed: '+path)
    for path, expected in contract['input_sha256'].items():
        if sha(OUT/path) != expected:
            raise ValueError('Frozen full-run input changed: '+path)
    if sha(BASE/'contract.json') != contract['baseline_contract_sha256']:
        raise ValueError('Baseline contract changed')
    if sha(CONFIG) != contract['cluster_config_sha256']:
        raise ValueError('Cluster configuration changed')
    if sha(ROOT/'data_reviews/listener_noise_rejections.json') != contract['mutable_source_feedback_sha256']:
        raise ValueError('Source manual feedback changed')
    return contract


def bind():
    global FEEDBACK, OLD_FEEDBACK
    full.OUT = OUT
    full.configure_normalization()
    cluster.OUT = OUT
    cluster.SCHED = SCHED
    cluster.SCRIPT = SCRIPT
    cluster.verify = verify
    FEEDBACK = read(OUT/'feedback.json')['rows']
    OLD_FEEDBACK = read(OUT/'qc4_review_feedback.json')['rows']


def initialize():
    if (OUT/'contract.json').exists():
        verify()
        return
    full.OUT = BASE
    ORIGINAL_VERIFY()
    latest = read(ROOT/'experiments/p3_listen_v8_audio_final_20261023/contract.json')
    for path, expected in latest['code_sha256'].items():
        assert sha(ROOT/path) == expected, path
    for folder in ('results','summaries','audit','errors','normalized_pairs','logs','manifests',
                   'cluster/claims','cluster/progress','cluster/nodes','cluster/logs','cluster/telemetry'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    inputs = ['catalog.json','human_rejections.json','human_retained.json','qc4_review_feedback.json']
    for name in inputs:
        shutil.copyfile(BASE/name, OUT/name)
    feedback = read(BASE/'qc4_review_feedback.json')['rows']
    feedback_sources = [ROOT/'data_reviews/listener_v8_recovery_feedback_20261023.json',
                        ROOT/'data_reviews/listener_v8_audio_first_feedback_20261023.json']
    for path in feedback_sources:
        feedback += read(path)['rows']
    write(OUT/'feedback.json', dict(rows=feedback, sources={str(p):sha(p) for p in feedback_sources}))
    inputs += ['feedback.json']
    rows = read(OUT/'catalog.json')['rows']
    assert len({r['pair'] for r in rows}) == len(rows)
    queue = [r['pair'] for r in rows]
    random.Random(2026102320).shuffle(queue)
    write(SCHED/'queue.json', dict(pairs=queue, seed=2026102320))
    write(SCHED/'queue_state.json', dict(cursor=0, total=len(queue)))
    inputs += ['cluster/queue.json']
    code = set(read(BASE/'contract.json')['code_sha256']) | set(latest['code_sha256']) | {
        'scripts/run_listen_v8_audio_full.py','scripts/plan_listen_v8_audio_full.py',
        'scripts/listen_v8_face_statistics.py','scripts/listen_v8_recovery.py',
        'scripts/run_listen_v8_cluster.py'}
    write(OUT/'contract.json', dict(revision=REVISION, utc=full.now(), audio_revision=policy.REVISION,
        audio_policy=policy.POLICY, segmentation_recovery_revision=recovery.REVISION,
        face_statistics_revision=faces.REVISION, baseline_contract_sha256=sha(BASE/'contract.json'),
        reviewed_audio_contract_sha256=sha(ROOT/'experiments/p3_listen_v8_audio_final_20261023/contract.json'),
        cluster_config_sha256=sha(CONFIG), code_sha256={p:sha(ROOT/p) for p in sorted(code)},
        input_sha256={p:sha(OUT/p) for p in inputs}, source_pairs=len(rows),
        mutable_source_feedback_sha256=sha(ROOT/'data_reviews/listener_noise_rejections.json'),
        source_scope=read(OUT/'catalog.json')['scope'], ast_batch=16, panns_batch=16,
        visual_filter_applied=False, orientation_filter_applied=False, render_media=False,
        face_measurement='native landmark extent, both roles; missing geometry explicitly counted',
        full_run_authorized=True, training_admitted=False,
        authorization='按备份版本全量切分+音质；先不做人脸区域裁剪筛选，统计人脸大小分布再确定阈值。'))
    write(SCHED/'control.json', dict(stop=False, drain=False, stage='fixed_4x2',
        nodes={n['name']:cluster.dimensions(4,2) for n in cluster.configuration()['nodes']}))
    write(OUT/'runtime.json', dict(state='ready', utc=full.now(), source_pairs=len(rows), training_started=False))
    verify()
    print(json.dumps(dict(initialized=str(OUT), pairs=len(rows))), flush=True)


def interval_key(row):
    return (row['speaker_id'], row['listener_id'], row['start'], row['end'])


def cache_matches(row, old, norm):
    if interval_key(row) != interval_key(old) or row['sources'] != old['sources']:
        return False
    if not all(k in old for k in ('qc3','qc4','normalized_audio')):
        return False
    return all(old['normalized_audio'][role]['sha256'] == norm['tracks'][row[role+'_index']]['sha256']
               for role in ('speaker','listener'))


def prepare(item):
    began = time.monotonic()
    contract = read(OUT/'contract.json')
    assert sha(ROOT/'data_reviews/listener_noise_rejections.json') == contract['mutable_source_feedback_sha256']
    try:
        audit = planner.inspect_pair(item)
    except ValueError as e:
        if not str(e).startswith(KNOWN_FAILURES):
            raise
        audit = dict(pair=item['pair'], clips=[], unresolved=[dict(reason=str(e))], ok=False, source_seconds=None)
    for row in audit['clips']:
        row['listener_screen'] = dict(accepted=True, deferred_reasons=
            ['listener_short_speech_unconfirmed'] if row['listener_compatibility']['uncertain_short_speech'] else [])
        row['duration_tier'] = '2to3s' if row['duration_s'] < 3 else 'over10s' if row['duration_s'] > 10 else 'ge3s'
    assert all(a['end'] <= b['start']+1e-6 for a,b in zip(audit['clips'], audit['clips'][1:]))
    write(OUT/'audit'/(item['pair']+'.json'), audit)
    norm = None
    cache = {}
    if audit['clips']:
        marker = OUT/'normalized_pairs'/(item['pair']+'.json')
        prior = BASE/'normalized_pairs'/marker.name
        if not marker.exists() and prior.exists():
            shutil.copyfile(prior, marker)
        norm = full.legacy.normalize(item, audit['clips'])
        for directory in (BASE, RECOVERY):
            path = directory/'results'/(item['pair']+'.json')
            if not path.exists():
                continue
            value = read(path)
            source_hash = sha(path)
            for old in value['clips']:
                cache[interval_key(old)] = (old, str(path), source_hash)
    return audit, norm, cache, time.monotonic()-began


def finish_row(plan, row, feedback):
    """Refresh task/human decisions for this plan; preserve waveform evidence."""
    row.update(plan)
    states = [v['status'] for v in row['qc3'].values()]
    automatic = 'exclude' if 'exclude' in states else 'review' if 'review' in states else 'keep'
    if plan['listener_screen']['deferred_reasons'] and automatic == 'keep':
        automatic = 'review'
    effective, hits = full.legacy.human_decision(plan, automatic,
        feedback['rejects'], feedback['positives'])
    row.update(automatic_decision=automatic, effective_decision=effective, human_decisions=hits,
        audio_qc_status=effective, v8_combined_status='exclude' if effective == 'exclude' or plan['source_manual_rejections']
        else 'review' if effective == 'review' or plan['boundary_review_flags'] else 'candidate',
        admission_pool='onset_reserve' if plan['boundary_activity']['onset_activity'] else 'primary')
    q = row['qc4']
    comp = plan['listener_compatibility']
    content = comp.get('uncertain_short_speech', []) + comp.get('substantive', [])
    q['task_reasons'] = ['listener_content_not_confirmed_backchannel'] if content else []
    q['content_events'] = content
    q['automatic_status'] = ('exclude' if row['v8_combined_status'] == 'exclude' else
        'review' if row['v8_combined_status'] == 'review' or q['task_reasons'] or q['audio_reasons'] else 'candidate')
    q.update(status=q['automatic_status'], original_qc3_status=effective,
             original_combined_status=row['v8_combined_status'], human_holds=[])
    row['release'] = original.release.route(row, OLD_FEEDBACK)
    row['audio_final'] = policy.route(row, FEEDBACK)
    row['recovery'] = recovery.analysis(row)
    row['training_admitted'] = False
    return row


def worker(node_name, local_rank):
    verify(); bind()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    node = cluster.node_config(node_name)
    rank = cluster.worker_rank(node['index'], local_rank)
    lookup = {r['pair']:r for r in read(OUT/'catalog.json')['rows']}
    queue = read(SCHED/'queue.json')['pairs']
    feedback = dict(rejects=read(OUT/'human_rejections.json')['rows'], positives=read(OUT/'human_retained.json')['rows'])
    counts = collections.Counter(); timing = collections.Counter()
    began = time.monotonic(); models = None; consecutive_errors = 0
    def progress(stage, pending=0, **kw):
        write(SCHED/'progress'/f'rank_{rank:03d}.json', dict(utc=full.now(), unix=time.time(), node=node_name,
            rank=rank, local_rank=local_rank, gpu=local_rank%8, pid=os.getpid(), stage=stage,
            pending_preparation=pending, counts=dict(counts), timing_seconds=dict(timing),
            wall_seconds=time.monotonic()-began, **kw))
    progress('starting')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {}
        def fill():
            control = cluster.read(SCHED/'control.json'); setting = control['nodes'][node_name]
            active = local_rank < setting['model_workers'] and not control.get('stop') and not control.get('drain')
            while active and len(pending) < min(2, setting['prep_per_worker']):
                pair = cluster.claim(queue, rank)
                if pair is None: break
                pending[pool.submit(prepare, lookup[pair])] = pair
                counts['claimed_pairs'] += 1
        while True:
            fill()
            if not pending:
                if cluster.exhausted() or cluster.read(SCHED/'control.json').get('stop'): break
                progress('idle_disabled'); time.sleep(2); continue
            done, _ = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                progress('preparing', len(pending)); continue
            for future in done:
                pair = pending.pop(future); fill(); progress('screening', len(pending), pair=pair)
                try:
                    audit, norm, cache, elapsed = future.result()
                    timing['preparation_sum'] += elapsed
                    clips = []
                    for plan in audit['clips']:
                        previous = cache.get(interval_key(plan))
                        if previous and cache_matches(plan, previous[0], norm):
                            row = copy.deepcopy(previous[0])
                            provenance = dict(reused=True, file=previous[1], sha256=previous[2], exact_interval=True,
                                              normalized_waveform_hashes_verified=True)
                            counts['cached_clips'] += 1
                        else:
                            if models is None:
                                progress('loading_models', len(pending), pair=pair)
                                models = (full.legacy.EventModel(full.legacy.MODEL, batch_size=16), full.legacy.PannsModel(batch_size=16))
                            row = RAW_SCREEN(plan, norm, *models, feedback['rejects'], feedback['positives'])
                            row = original.assess(row, OLD_FEEDBACK)
                            provenance = dict(reused=False, algorithm='frozen QC3 + QC4', normalized_waveform_hashes_verified=True)
                            counts['fresh_clips'] += 1
                        row = finish_row(plan, row, feedback)
                        row['quality_provenance'] = provenance
                        clips.append(row)
                    result = dict(pair=pair, split=lookup[pair]['split'], group_id=lookup[pair]['group_id'],
                        complete=True, utc=full.now(), revision=REVISION, clips=clips, rejected=audit['unresolved'],
                        source_ok=audit['ok'], source_seconds=audit.get('source_seconds'), training_ready=False,
                        visual_filter_applied=False, scheduler=dict(node=node_name, rank=rank))
                    write(OUT/'results'/(pair+'.json'), result)
                    write_pair_summary(result)
                    counts['complete_pairs'] += 1; counts['clips'] += len(clips); consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1; counts['error_pairs'] += 1
                    write(OUT/'errors'/(pair+'.json'), dict(pair=pair, utc=full.now(), error=repr(e),
                        traceback=traceback.format_exc(), node=node_name, rank=rank))
                    if consecutive_errors >= 10:
                        raise RuntimeError('Ten consecutive processing errors; stopped worker') from e
                progress('pair_finished', len(pending), pair=pair)
    progress('finished')


def write_pair_summary(result):
    value = dict(pair=result['pair'], source_ok=result['source_ok'], source_seconds=result['source_seconds'],
        pools=faces.summarize_clips(result['clips']),
        unresolved=dict(collections.Counter(str(r.get('reason','unknown')) for r in result['rejected'])))
    write(OUT/'summaries'/(result['pair']+'.json'), value)


def report(export=False):
    with (OUT/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = {r['pair'] for r in read(OUT/'catalog.json')['rows']}
        pools = {}; done = set(); failed_sources = 0; source_seconds = 0.; reasons = collections.Counter()
        for path in (OUT/'summaries').glob('*.json'):
            r = read(path); assert r['pair'] in expected
            done.add(r['pair']); faces.merge_counts(pools, r['pools']); reasons.update(r['unresolved'])
            failed_sources += not r['source_ok']; source_seconds += r['source_seconds'] or 0.
        errors = {p.stem for p in (OUT/'errors').glob('*.json')} - done
        summary = dict(utc=full.now(), revision=REVISION, total_pairs=len(expected), complete_pairs=len(done),
            error_pairs=len(errors), pending_pairs=len(expected-done-errors), source_invalid_pairs=failed_sources,
            covered_source_hours=source_seconds/3600, counts={k:v['clips'] for k,v in pools.items()},
            hours={k:v['seconds']/3600 for k,v in pools.items()},
            cached_clips=sum(v['cached_clips'] for v in pools.values()), fresh_clips=sum(v['fresh_clips'] for v in pools.values()),
            unresolved_reasons=dict(reasons), visual_filter_applied=False, rendering_deferred=True,
            training_admitted=False, final=bool(export),
            note='Disjoint intervals per pair; roles/splits/onset reserve remain separate. Partial until all sources accounted for. Errors are not audio rejections.')
        distribution = dict(utc=summary['utc'], revision=faces.REVISION, complete_pairs=len(done), total_pairs=len(expected),
            final=bool(export), pools=pools, pixel_edges=[None if x==float('inf') else x for x in faces.PIXEL_EDGES],
            area_fraction_edges=faces.AREA_EDGES, validity_edges=[0,.5,.8,.9,.99,1.01],
            definitions=dict(median_bins='One clip per bin using median valid-frame native landmark extent; seconds sum clip durations.',
                frame_bins='Valid frame counts, no upsampling or crop expansion.',
                threshold_seconds='Exploratory p10 short-side >= threshold among valid frames; NOT admission, inspect validity separately.',
                unmeasured='Included in audio pools and reported separately, never counted as zero-size face.'),
            visual_filter_applied=False)
        write(OUT/'screening_summary.json', summary)
        write(OUT/'face_size_distribution.json', distribution)
        if export:
            handles = {}
            try:
                for pair in sorted(done):
                    result_path = OUT/'results'/(pair+'.json')
                    for row in read(result_path)['clips']:
                        key = '_'.join((row['split'], row['admission_pool'], row['audio_final']['pool']))
                        if key not in handles: handles[key] = (OUT/'manifests'/(key+'.jsonl.pending')).open('w')
                        compact = {k:v for k,v in row.items() if k not in ('qc3','qc4')}
                        compact['quality_evidence_file'] = str(result_path)
                        handles[key].write(json.dumps(compact, ensure_ascii=False)+'\n')
            finally:
                for f in handles.values(): f.close()
            for key in handles:
                (OUT/'manifests'/(key+'.jsonl.pending')).replace(OUT/'manifests'/(key+'.jsonl'))
            write(OUT/'manifests/index.json', dict(summary, manifest_sha256={k:sha(OUT/'manifests'/(k+'.jsonl')) for k in handles}))
        return summary


def supervise():
    verify(); bind()
    cfg = cluster.configuration()
    with (SCHED/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        reporter = None; last_report = 0; began = time.time()
        try:
            for node in cfg['nodes']:
                print(cluster.ssh(node, [cfg['python'], str(SCRIPT), 'detach-node', '--node', node['name']]), flush=True)
            while True:
                snap = cluster.progress(); nodes = snap['nodes']; now = time.time()
                if len(nodes)==len(cfg['nodes']) and all(n['state']=='finished' for n in nodes.values()): break
                for name, node in nodes.items():
                    if node['state']=='failed': raise RuntimeError(name+' worker failure')
                    if now-node['unix']>120: raise RuntimeError(name+' stale heartbeat')
                    pressure = cluster.pressure(node)
                    if pressure:
                        ctrl = cluster.read(SCHED/'control.json')
                        desired = cluster.dimensions(1,1)
                        if ctrl['nodes'][name] != desired:
                            ctrl['nodes'][name] = desired; ctrl['stage'] = 'resource_backoff'
                            write(SCHED/'control.json', ctrl)
                            with (SCHED/'safety_events.jsonl').open('a') as f:
                                f.write(json.dumps(dict(utc=full.now(), node=name, reasons=pressure, setting=desired))+'\n')
                if now-began>120 and len(nodes)!=len(cfg['nodes']): raise RuntimeError('Missing node heartbeat')
                if reporter is not None and reporter.poll() not in (None,0): raise RuntimeError('Statistics reporter failed')
                if now-last_report>300 and (reporter is None or reporter.poll() is not None):
                    with (OUT/'logs/report.log').open('ab', buffering=0) as log:
                        reporter = subprocess.Popen([sys.executable,'-u',str(SCRIPT),'report'], stdout=log, stderr=log, stdin=subprocess.DEVNULL)
                    last_report = now
                write(OUT/'runtime.json', dict(state='cluster_screening', utc=full.now(), coordinator_pid=os.getpid(),
                    progress=snap['counts'], queue=snap['queue'], nodes={k:n['state'] for k,n in nodes.items()},
                    control=cluster.read(SCHED/'control.json'), visual_filter_applied=False, training_started=False))
                write(SCHED/'latest.json', snap)
                time.sleep(10)
            if reporter: reporter.wait()
            # Repair only a missing lightweight summary after a process interruption.
            for path in (OUT/'results').glob('*.json'):
                if not (OUT/'summaries'/path.name).exists(): write_pair_summary(read(path))
            summary = report(export=True)
            assert summary['pending_pairs']==0, 'Unaccounted queue claims; resume required'
            write(OUT/'runtime.json', dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',
                utc=full.now(), summary=summary, workers=0, visual_filter_applied=False, training_started=False))
        except BaseException as e:
            ctrl = cluster.read(SCHED/'control.json'); ctrl['stop'] = True; write(SCHED/'control.json',ctrl)
            write(OUT/'runtime.json', dict(state='stopping_after_error', utc=full.now(), error=repr(e), training_started=False))
            raise


def resume():
    """Restart only after all previous run-owned workers have exited."""
    verify(); bind()
    code = """from pathlib import Path
import json
script = %r
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=p.read_bytes().decode().split('\\0')
    except (OSError,UnicodeError): continue
    if script in args and any(s in args for s in ('worker','launch-node','supervise')):
        active.append(int(p.parent.name))
print(json.dumps(active))
""" % str(SCRIPT)
    for node in cluster.configuration()['nodes']:
        active = json.loads(cluster.ssh(node, [cluster.configuration()['python'], '-c', code]))
        if active: raise RuntimeError('Run-owned processes still active on '+node['name']+': '+str(active))
    # Keep the immutable queue, all results and errors; only reset its cursor.
    attempt = SCHED/'attempts'/str(time.time_ns())
    for folder in ('nodes','progress'):
        old = SCHED/folder
        attempt.mkdir(parents=True, exist_ok=True)
        old.rename(attempt/folder)
        old.mkdir()
    for path in (OUT/'results').glob('*.json'):
        if not (OUT/'summaries'/path.name).exists(): write_pair_summary(read(path))
    write(SCHED/'queue_state.json', dict(cursor=0,total=len(read(SCHED/'queue.json')['pairs'])))
    write(SCHED/'control.json', dict(stop=False,drain=False,stage='fixed_4x2',
        nodes={n['name']:cluster.dimensions(4,2) for n in cluster.configuration()['nodes']}))
    detach()


def detach():
    verify(); bind()
    with (SCHED/'logs/coordinator.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable,'-u',str(SCRIPT),'supervise'], stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True)
    write(SCHED/'launch.json', dict(utc=full.now(), pid=process.pid))
    print(json.dumps(read(SCHED/'launch.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init','detach','resume','supervise','worker','launch-node','detach-node','report','verify'])
    parser.add_argument('--node'); parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.stage == 'init': initialize()
    elif args.stage == 'verify': print(json.dumps(verify(), ensure_ascii=False)[:500])
    elif args.stage == 'report': print(json.dumps(report(args.export), ensure_ascii=False), flush=True)
    elif args.stage == 'detach': detach()
    elif args.stage == 'resume': resume()
    elif args.stage == 'supervise': supervise()
    else:
        bind()
        if args.stage == 'worker': worker(args.node, args.local_rank)
        elif args.stage == 'launch-node': cluster.launch_node(args.node)
        elif args.stage == 'detach-node': cluster.detach_node(args.node)
