"""Resumable V6/QC3 corpus screening only; never export caches or start training.

The historical 300h manifest is not an input. Only its source catalog and
verified whole-track normalization may be reused. Every clip is replanned.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import preview_listen_v6 as preview
from p3_listen.common import read, write, write_once, sha, digest
from p3_listen import subturns
from listen_policy_v6 import POLICY as VIDEO_POLICY, reviewed_noise_hits, seconds
from listen_audio_quality import EventModel, assess
from listen_audio_quality_v3 import PannsModel, scan_track, prepare_pair, POLICY
from screen_listen_qc_sample import annotations, metadata, read_wave, MODEL

ROOT = preview.ROOT
OUT = ROOT / 'data_corpora/seamless_listen_qc3_full'
CODE = ['scripts/run_listen_qc3_bulk.py', 'scripts/preview_listen_v6.py',
        'scripts/listen_policy_v6.py', 'scripts/build_seamless_rule_turns.py',
        'scripts/prepare_listen_300h.py', 'scripts/listen_audio_quality.py',
        'scripts/listen_audio_quality_v3.py', 'scripts/screen_listen_qc_sample.py',
        'scripts/build_seamless_listening_preview.py', 'src/p3_listen/subturns.py',
        'src/p3_listen/media.py', 'src/p3_listen/common.py']


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def human_decision(row, automatic, rejects, positives):
    """Negative intervals survive recutting; positive overrides require exact clips."""
    hits = reviewed_noise_hits(row['listener_id'], row['start'], row['end'], rejects)
    if hits:
        return 'exclude', hits
    hits = [r for r in positives if r['pair'] == row['pair']
            and r['listener_id'] == row['listener_id']
            and abs(r['start']-row['start']) < 1e-6
            and abs(r['end']-row['end']) < 1e-6]
    return ('keep', hits) if hits else (automatic, [])


def init():
    if (OUT/'contract.json').exists():
        verify_contract()
        print('Existing frozen run; resume with launch-node', flush=True)
        return
    catalog = read(preview.OLD/'catalog.json')
    if sha(catalog['index']) != catalog['index_sha256']:
        raise ValueError('Original source index changed; rebuild isolated catalog first')
    # Same participant/session isolation as the existing source catalog; test excluded.
    rows = catalog['train'] + catalog['dev']
    assert all(r['split'] in ('train', 'dev') for r in rows)
    assert not ({r['group_id'] for r in catalog['train']} &
                {r['group_id'] for r in catalog['dev']})
    selected = {r['pair'] for r in read(preview.OLD/'selected.json')['pairs']}
    rows.sort(key=lambda r: (r['pair'] not in selected, digest(r['pair'])))
    write_once(OUT/'catalog.json', dict(rows=rows, scope=catalog['policy'],
        source_index=catalog['index'],source_index_sha256=catalog['index_sha256'],
        excluded_pairs=catalog['excluded_pairs']))
    rejects = read(ROOT/'data_reviews/listener_noise_rejections.json')
    positives = {}
    for filename in ['listener_qc1_feedback.json', 'listener_qc3_feedback.json']:
        for r in read(ROOT/'data_reviews'/filename)['rows']:
            if r['decision'] == 'retain_valid_listener_response':
                positives[r['sample_id']] = r
    write_once(OUT/'human_rejections.json', rejects)
    write_once(OUT/'human_retained.json', dict(rows=list(positives.values())))
    contract = dict(revision='v6-qc3-full-1', utc=now(), video_policy=VIDEO_POLICY,
        audio_policy=POLICY, normalization_policy=subturns.POLICY,
        scope='all isolated official train/dev source pairs; no test or old training windows',
        nodes=3, workers_per_node=8, world_size=24, ast_batch=16, panns_batch=16,
        code_sha256={f:sha(ROOT/f) for f in CODE},
        input_sha256={f:sha(OUT/f) for f in
                      ['catalog.json','human_rejections.json','human_retained.json']},
        models={name:read(ROOT/'models'/name/'provenance.json')
                for name in ['audio_qc_ast','audio_qc_panns']},
        admission='keep is a candidate; review quarantined; exclude retained for audit',
        training_authorized=False, training_ready=False)
    write_once(OUT/'contract.json', contract)
    print(dict(pairs=len(rows), splits=dict(Counter(r['split'] for r in rows)),
               contract=str(OUT/'contract.json')), flush=True)


def verify_contract():
    contract = read(OUT/'contract.json')
    for f,h in contract['code_sha256'].items():
        if sha(ROOT/f) != h:
            raise ValueError('Frozen screening code changed: '+f)
    for f,h in contract['input_sha256'].items():
        if sha(OUT/f) != h:
            raise ValueError('Frozen input changed: '+f)
    return contract


def normalize(item, clips):
    import soundfile as sf
    path = OUT/'normalized_pairs'/(item['pair']+'.json')
    old = preview.OLD/'normalized_pairs'/(item['pair']+'.json')
    for candidate in (path, old):
        if not candidate.exists():
            continue
        value = read(candidate)
        if value.get('summary', {}).get('ok') and len(value.get('tracks', [])) == 2:
            if all(Path(t['path']).exists() and sha(t['path']) == t['sha256']
                   for t in value['tracks']):
                return value
    # Source .wav, full-track fixed gain, same -18 LUFS policy as the reviewed page.
    tracks = []
    row = clips[0]
    for index in (0, 1):
        role = 'speaker' if row['speaker_index'] == index else 'listener'
        source = row['sources'][role]
        meta = metadata(source['archive'], source['member'])
        speech = seconds([[v['start'], v['end']] for v in meta.get('metadata:vad', [])])
        value = subturns.normalize(item['label']+'_'+item['ab'[index]]['id'], source, speech)
        # Verify exact original audio length and output format, even for silent sources.
        with tarfile.open(source['archive'],'r:') as archive:
            with archive.extractfile(str(Path(source['member']).with_suffix('.wav'))) as wav:
                with sf.SoundFile(wav) as original:
                    expected = round(len(original)*48000/original.samplerate)
        info = sf.info(value['path'])
        if info.samplerate != 48000 or info.channels != 1 or info.frames != expected:
            raise ValueError('Normalization changed audio geometry')
        if speech >= 2 and not value['target_reached']:
            raise ValueError('Normalization target not reached')
        if value['after']['input_tp'] is not None and value['after']['input_tp'] > -.8:
            raise ValueError('Normalized true peak exceeds tolerance')
        tracks.append(value)
    value = dict(summary=dict(ok=True,pair=item['pair']), tracks=tracks)
    write(path, value)
    return value


def prepare(item):
    if sha(ROOT/'data_reviews/listener_noise_rejections.json') != sha(OUT/'human_rejections.json'):
        raise ValueError('Manual rejection list changed during frozen run')
    audit = preview.inspect_pair(item, defer_audio_uncertainty=True)
    # The preview function reads the current manual list. The run contract checks
    # that it still matches the frozen copy before any planning is performed.
    intervals=sorted((r['start'],r['end']) for r in audit['clips'])
    if any(b > c+1e-7 for (_,b),(c,_) in zip(intervals,intervals[1:])):
        raise ValueError('Overlapping canonical listening clips')
    norm = normalize(item, audit['clips']) if audit['clips'] else None
    return audit, norm


def scan(row, norm, ast, panns, rejects, positives):
    waves = {role:read_wave(norm['tracks'][row[role+'_index']]['path'],
                           row['start'], row['duration_s']) for role in ('speaker','listener')}
    paired = prepare_pair(waves['speaker'], waves['listener'])
    checks = {}
    for role in ('listener','speaker'):
        src = row['sources'][role]
        vad,words = annotations(metadata(src['archive'],src['member']),row['start'],row['end'])
        old = assess(waves[role],48000,vad,words,ast,role=role)
        checks[role] = scan_track(waves[role],old,panns,
                                  paired=paired if role=='listener' else None,role=role)
    states = [v['status'] for v in checks.values()]
    automatic = 'exclude' if 'exclude' in states else 'review' if 'review' in states else 'keep'
    if row['listener_screen'].get('deferred_reasons') and automatic=='keep':
        automatic = 'review'
    effective,hits = human_decision(row,automatic,rejects,positives)
    return dict(row, sample_id=row['sample_id'].replace('_v6_','_qc3full_'),
        qc3=checks, automatic_decision=automatic, effective_decision=effective,
        human_decisions=hits, listener_human_evidence=checks['listener']['has_human_evidence'],
        normalized_audio={role:norm['tracks'][row[role+'_index']] for role in ('speaker','listener')},
        training_admitted=False)


def worker(rank, limit=0):
    import torch
    torch.set_num_threads(2)
    contract = verify_contract()
    # Do not silently change manual exclusions during an active frozen run.
    if sha(ROOT/'data_reviews/listener_noise_rejections.json') != sha(OUT/'human_rejections.json'):
        raise ValueError('Manual rejection list changed; use a new frozen run')
    preview.OUT = OUT
    subturns.OUT = OUT
    rows = read(OUT/'catalog.json')['rows'][rank::contract['world_size']]
    if limit:
        rows = rows[:limit]
    rejects = read(OUT/'human_rejections.json')['rows']
    positives = read(OUT/'human_retained.json')['rows']
    progress = OUT/'progress'/f'rank_{rank:02d}.json'
    counts,hours = Counter(),Counter()
    def report(stage, **kw):
        value = dict(utc=now(),pid=os.getpid(),rank=rank,total_pairs=len(rows),
                     stage=stage,counts=dict(counts),candidate_hours=dict(hours),**kw)
        write(progress,value)
        print(json.dumps(value),flush=True)
    report('loading_models')
    ast = EventModel(MODEL,batch_size=contract['ast_batch'])
    panns = PannsModel(batch_size=contract['panns_batch'])
    consecutive_errors = 0
    pending = []
    for item in rows:
        marker = OUT/'results'/(item['pair']+'.json')
        if marker.exists() and read(marker).get('complete'):
            counts['previously_complete_pairs'] += 1
        else:
            pending.append(item)
    # One CPU prefetch per GPU overlaps source inspection/normalization and inference.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare,pending[0]) if pending else None
        for index,item in enumerate(pending):
            report('screening_pair',pair=item['pair'],index=index)
            try:
                try:
                    audit,norm = future.result()
                finally:
                    future = pool.submit(prepare,pending[index+1]) if index+1<len(pending) else None
                results=[]
                if not audit['ok']:
                    reasons=[r.get('reason','') for r in audit['rejected']]
                    # Geometry/source validation failures remain auditable, not qualified data.
                    allowed=('source_orientation_not_upright','orientation_ambiguous',
                        'orientation_insufficient_landmarks','orientation_invalid_keypoint_shape',
                        'Source AV/keypoint length disagreement','Missing transcript')
                    if any(not str(r).startswith(allowed) for r in reasons):
                        raise RuntimeError(str(reasons))
                for row in audit['clips']:
                    value=scan(row,norm,ast,panns,rejects,positives)
                    results.append(value)
                write(OUT/'results'/(item['pair']+'.json'),dict(pair=item['pair'],
                    split=item['split'],group_id=item['group_id'],complete=True,utc=now(),
                    clips=results,rejected=audit['rejected'],source_ok=audit['ok']))
                for value in results:
                    key=item['split']+'/'+value['effective_decision']
                    counts[key]+=1
                    hours[key]+=value['unique_seconds']/3600
                counts['complete_pairs']+=1
                consecutive_errors=0
            except Exception as e:
                # Errors remain retryable; never count as screened or passed.
                counts['error_pairs']+=1
                consecutive_errors+=1
                write(OUT/'errors'/(item['pair']+'.json'),dict(pair=item['pair'],utc=now(),
                    error=type(e).__name__+': '+str(e)))
                report('pair_error',pair=item['pair'],error=str(e))
                if consecutive_errors>=10:
                    raise RuntimeError('Ten consecutive pair failures; stopping this worker') from e
            report('pair_finished',pair=item['pair'],index=index)
    report('finished_with_errors' if counts['error_pairs'] else 'finished')


def summarize():
    """Audit all completed pair records; count no candidate as training-ready."""
    catalog=read(OUT/'catalog.json');counts=Counter();durations=Counter();media_hours=Counter()
    reasons=Counter();complete=0;failed=0;source_rejected=0;voice=Counter()
    for item in catalog['rows']:
        marker=OUT/'results'/(item['pair']+'.json')
        if not marker.exists():
            failed+=int((OUT/'errors'/(item['pair']+'.json')).exists())
            continue
        result=read(marker)
        if not result.get('complete'):
            continue
        complete+=1;source_rejected+=int(not result['source_ok'])
        for r in result['rejected']:
            values=r.get('reason',r.get('rejection','unknown'))
            for reason in values if isinstance(values,list) else [values]:
                reasons[reason]+=1
        for r in result['clips']:
            key=r['split']+'/'+r['effective_decision']
            counts[key]+=1;durations[key]+=r['unique_seconds']/3600
            media_hours[key]+=r['duration_s']/3600
            if r['listener_human_evidence']:
                voice[key]+=1
    total=len(catalog['rows'])
    value=dict(utc=now(),total_pairs=total,complete_pairs=complete,error_pairs=failed,
        pending_pairs=total-complete-failed,source_rejected_pairs=source_rejected,
        counts=dict(counts),unique_hours=dict(durations),media_hours=dict(media_hours),
        with_model_vocal_evidence_counts=dict(voice),geometry_boundary_rejection_reasons=dict(reasons),
        catalog_excluded_pairs=catalog['excluded_pairs'],training_ready=False,
        note='keep remains candidate pending aggregate audit/spot check; review excluded from training. '
             'Unique hours count nonoverlapping speaker spans including internal pauses, excluding frame-alignment silence.')
    write(OUT/'summary.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return value


def launch_node(node):
    contract=verify_contract()
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    with (OUT/f'node_{node}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        procs=[]
        for gpu in range(8):
            rank=node*8+gpu
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            with (OUT/'logs'/f'rank_{rank:02d}.log').open('ab',buffering=0) as log:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    'worker','--rank',str(rank)],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,env=env)
            procs.append((rank,p))
        write(OUT/f'node_{node}.json',dict(node=node,utc=now(),supervisor_pid=os.getpid(),
              state='running',workers=[dict(rank=r,pid=p.pid) for r,p in procs]))
        codes={str(r):p.wait() for r,p in procs}
        write(OUT/f'node_{node}.json',dict(node=node,utc=now(),state='finished',exit_codes=codes))
        # Last completing node publishes the authoritative aggregate once.
        with (OUT/'summary.lock').open('a') as summary_lock:
            fcntl.flock(summary_lock,fcntl.LOCK_EX)
            if all((OUT/f'node_{n}.json').exists() and
                   read(OUT/f'node_{n}.json').get('state')=='finished' for n in range(3)):
                summarize()
        if any(codes.values()):
            raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['init','worker','launch-node','summary'])
    p.add_argument('--rank',type=int,choices=range(24),default=0)
    p.add_argument('--node',type=int,choices=range(3),default=0)
    p.add_argument('--limit',type=int,default=0,help='smoke test only; default entire shard')
    a=p.parse_args()
    if a.stage=='init':init()
    elif a.stage=='worker':worker(a.rank,a.limit)
    elif a.stage=='summary':summarize()
    else:launch_node(a.node)
