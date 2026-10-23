"""Three-node full-source listening recovery, frozen audio QC, crop statistics.

Separate resumable queue and results. Parent candidates are archived separately
from terminal leaves so accepted child time is never counted twice.
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
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_audio_full as baseline
import run_listen_v8_audio_governor as governor_module
import listen_local_recovery as policy
import listen_full_source_runtime as acceleration
import listen_crop_distribution as crops
import listen_local_health as health
import listen_local_export as manifests
from p3_listen.common import read, sha
import numpy as np


def json_scalar(value):
    """Preserve numeric/boolean evidence types emitted by NumPy-based QC."""
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported JSON value: '+type(value).__name__)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,ensure_ascii=False,indent=2,default=json_scalar)+'\n'
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(encoded);temporary.replace(path)

ROOT=baseline.ROOT
BASE=baseline.OUT
OUT=ROOT/'data_corpora/seamless_listen_local_full'
SCHED=OUT/'cluster'
SCRIPT=Path(__file__).resolve()
CONFIG=ROOT/'configs/p3_listen_local_full_cluster.json'
REVISION='seamless-full-source-no-scene-crop-statistics-2'
cluster=baseline.cluster
full=baseline.full
faces=baseline.faces
BASE_BIND=baseline.bind
BASE_REPORT=baseline.report


def dimensions(models,prep):
    cfg=cluster.configuration()
    if models not in (1,2,4,8,16) or prep not in (1,2) or models*8>cfg['maximum_model_workers_per_node'] or models*8*prep>cfg['maximum_preparation_slots_per_node']:
        raise ValueError('Unsafe scheduler dimensions')
    return dict(models_per_gpu=models,prep_per_worker=prep,model_workers=8*models,preparation_slots=8*models*prep)


def initial_nodes():
    return {n['name']:dimensions(*n.get('initial_setting',[4,2])) for n in cluster.configuration()['nodes']}


def rank_enabled(node,local_rank,setting):
    limits=node.get('gpu_model_limits',[node.get('maximum_models_per_gpu',4)]*8)
    return local_rank<setting['model_workers'] and local_rank//8<limits[local_rank%8]


def source_context(item):
    # User correction: Seamless participant streams do not require RealTalk's
    # shot scan. Keep full word/role traversal and all acoustic/QC constraints.
    return policy.SourceContext(item,shot_times=[[],[]])


def verify():
    contract=read(OUT/'contract.json')
    for p,digest in contract['code_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Frozen code changed: '+p)
    for p,digest in contract['input_sha256'].items():
        if sha(OUT/p)!=digest:raise ValueError('Frozen input changed: '+p)
    assert sha(BASE/'contract.json')==contract['baseline_contract_sha256']
    assert sha(CONFIG)==contract['cluster_config_sha256']
    assert sha(ROOT/'data_reviews/listener_noise_rejections.json')==contract['source_feedback_sha256']
    return contract


def bind():
    baseline.OUT=OUT;baseline.SCHED=SCHED;baseline.SCRIPT=SCRIPT;baseline.REVISION=REVISION
    BASE_BIND()
    cluster.CONFIG=CONFIG;cluster.SCRIPT=SCRIPT;cluster.verify=verify
    cluster.dimensions=dimensions;governor_module.dimensions=dimensions
    health.install(cluster)
    governor_module.run=sys.modules[__name__]


def initialize():
    if (OUT/'contract.json').exists():verify();return
    baseline.verify()
    for d in ('results','summaries','audit','errors','normalized_pairs','normalized_audio','scene_cache','logs','manifests',
              'cluster/claims','cluster/progress','cluster/nodes','cluster/logs','cluster/telemetry','cluster/governor'):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    inputs=['catalog.json','human_rejections.json','human_retained.json','qc4_review_feedback.json','feedback.json']
    for p in inputs:shutil.copyfile(BASE/p,OUT/p)
    rows=read(OUT/'catalog.json')['rows'];queue=[r['pair'] for r in rows]
    assert len(set(queue))==len(queue)
    random.Random(2026102316).shuffle(queue)
    write(SCHED/'queue.json',dict(pairs=queue,seed=2026102316));inputs.append('cluster/queue.json')
    write(SCHED/'queue_state.json',dict(cursor=0,total=len(queue)))
    code=set(read(BASE/'contract.json')['code_sha256'])|{
        'scripts/listen_local_recovery.py','scripts/listen_full_source_runtime.py',
        'scripts/listen_crop_distribution.py','scripts/run_listen_local_full.py',
        'scripts/run_listen_v8_audio_governor.py','scripts/listen_local_health.py','scripts/listen_local_export.py'}
    write(OUT/'contract.json',dict(revision=REVISION,utc=full.now(),source_pairs=len(rows),
        source_scope=read(OUT/'catalog.json')['scope'],segmentation_revision=policy.REVISION,
        audio_revision=baseline.policy.REVISION,acceleration_revision=acceleration.REVISION,
        crop_statistics_revision=crops.REVISION,code_sha256={p:sha(ROOT/p) for p in sorted(code)},
        input_sha256={p:sha(OUT/p) for p in inputs},baseline_contract_sha256=sha(BASE/'contract.json'),
        cluster_config_sha256=sha(CONFIG),source_feedback_sha256=sha(ROOT/'data_reviews/listener_noise_rejections.json'),
        ast_batch=16,panns_batch=16,visual_filter_applied=False,render_media=False,
        accepted_parent_child_double_counting=False,training_admitted=False,
        user_authorization='用户认可两条连续提取小批，要求备份后全量处理 Seamless、统计面部裁剪框分布，并用满三机24卡安全提速。'))
    bind()
    write(SCHED/'control.json',dict(stop=False,drain=False,stage='fixed_4x2',
        nodes=initial_nodes()))
    write(OUT/'runtime.json',dict(state='ready',utc=full.now(),source_pairs=len(rows),training_started=False))
    verify()


def prepare(item):
    began=time.monotonic()
    path=BASE/'results'/(item['pair']+'.json')
    previous=read(path) if path.exists() else dict(clips=[])
    try:context=source_context(item)
    except ValueError as e:
        if not str(e).startswith(baseline.KNOWN_FAILURES):raise
        return dict(context=None,candidates=[],norm=None,cache={},elapsed=time.monotonic()-began,
                    diagnostic=dict(blocked=[],unresolved=[dict(reason=str(e))]))
    candidates,diagnostic=context.propose(previous['clips'])
    norm=None;cache={}
    if candidates:
        marker=OUT/'normalized_pairs'/(item['pair']+'.json');prior=BASE/'normalized_pairs'/marker.name
        if not marker.exists() and prior.exists():shutil.copyfile(prior,marker)
        norm=full.legacy.normalize(item,candidates)
        if path.exists():
            digest=sha(path)
            cache={baseline.interval_key(r):(r,str(path),digest) for r in previous['clips']}
        pilot=ROOT/'experiments/p3_listen_full_source_recovery_20261023/results'/(item['pair']+'.json')
        if pilot.exists():
            digest=sha(pilot)
            for r in read(pilot)['attempts']:cache[baseline.interval_key(r)]=(r,str(pilot),digest)
    return dict(context=context,candidates=candidates,diagnostic=diagnostic,norm=norm,cache=cache,
                old_kept=[(r['start'],r['end']) for r in previous['clips'] if r['audio_final']['pool']=='keep_candidate'],
                elapsed=time.monotonic()-began)


def screen_prepared(item,prepared,models,feedback,progress=lambda **kw:None):
    context=prepared['context'];diagnostic=prepared['diagnostic'];norm=prepared['norm'];cache=prepared['cache']
    pending=[(r,0,None) for r in prepared['candidates']];seen=set();leaves=[];parents=[]
    counts=collections.Counter();started=time.monotonic()
    while pending:
        plan,depth,parent=pending.pop(0);key=baseline.interval_key(plan)
        if key in seen:continue
        seen.add(key);plan.update(recovery_depth=depth,recovery_parent_id=parent)
        old=cache.get(key)
        if old and baseline.cache_matches(plan,old[0],norm):
            row=copy.deepcopy(old[0]);provenance=dict(reused=True,file=old[1],sha256=old[2],exact_interval=True,
                                                    normalized_waveform_hashes_verified=True)
            counts['cached_clips']+=1
        else:
            if not models:
                progress(stage='loading_models',pair=item['pair'])
                models.extend([full.legacy.EventModel(full.legacy.MODEL,batch_size=16),full.legacy.PannsModel(batch_size=16)])
            row=baseline.RAW_SCREEN(plan,norm,*models,feedback['rejects'],feedback['positives'])
            row=baseline.original.assess(row,baseline.OLD_FEEDBACK)
            provenance=dict(reused=False,fresh_qc=True,algorithm='frozen QC3 + QC4',normalized_waveform_hashes_verified=True)
            counts['fresh_clips']+=1
        row=baseline.finish_row(plan,row,feedback);row['quality_provenance']=provenance
        children=[]
        if row['audio_final']['pool']!='keep_candidate':
            blocks,kind=policy.quality_blocks(row)
            if kind=='localized_quality' and depth<6:
                children,detail=context.propose([row],domain=row)
                diagnostic['blocked'].extend(detail['blocked']);diagnostic['unresolved'].extend(detail['unresolved'])
                children=[c for c in children if baseline.interval_key(c) not in seen]
                for child in children:
                    assert child['duration_s']<row['duration_s']-1e-5
                    pending.append((child,depth+1,row['sample_id']))
            elif kind=='localized_quality':
                diagnostic['unresolved'].append(dict(start=row['start'],end=row['end'],reason='recovery_depth_limit'))
        (parents if children else leaves).append(row)
        counts['attempted_clips']+=1
        progress(stage='screening',pair=item['pair'],attempted_clips=counts['attempted_clips'],remaining_clips=len(pending))
    leaves.sort(key=lambda r:(r['start'],r['end']))
    assert all(a['end']<=b['start']+1e-6 for a,b in zip(leaves,leaves[1:]))
    for row in leaves:
        assert not any(row['start']<t<row['end'] for t in context.shots)
        row['crop_proposals']={role:crops.measure(context.geometry[index],row['start'],row['end'])
                               for role,index in [('speaker',row['speaker_index']),('listener',row['listener_index'])]}
    kept=[(r['start'],r['end']) for r in leaves if r['audio_final']['pool']=='keep_candidate']
    old_kept=prepared.get('old_kept',[])
    result=dict(pair=item['pair'],split=item['split'],group_id=item['group_id'],complete=True,utc=full.now(),
        revision=REVISION,scene_detection_applied=False,clips=leaves,superseded_attempts=parents,rejected=diagnostic['unresolved'],
        source_ok=context is not None,source_seconds=context.limit if context else None,
        source_shots=context.shots_by_track if context else None,training_ready=False,visual_filter_applied=False,
        counts=dict(counts),timing_seconds=dict(preparation=prepared['elapsed'],quality_and_statistics=time.monotonic()-started),
        coverage=dict(old_kept_seconds=policy.seconds(old_kept),new_kept_seconds=policy.seconds(kept),
            added_seconds=policy.seconds(policy.subtract(kept,old_kept)),removed_seconds=policy.seconds(policy.subtract(old_kept,kept))))
    audit=dict(pair=item['pair'],source_seconds=result['source_seconds'],diagnostic=diagnostic,
        shots=result['source_shots'],regions=context.regions if context else [],source_gaps=context.source_gaps if context else [],
        terminal_clips=[{k:r[k] for k in ('sample_id','start','end','speaker_index')} for r in leaves],
        retained_intervals=kept,parent_candidates_not_counted_in_terminal_pools=[r['sample_id'] for r in parents])
    return result,audit


def write_pair_summary(result):
    pools=faces.summarize_clips(result['clips']);proposal=crops.summarize(result['clips'])
    for key,value in proposal.items():pools[key]['crop_roles']=value
    write(OUT/'summaries'/(result['pair']+'.json'),dict(pair=result['pair'],source_ok=result['source_ok'],
        source_seconds=result['source_seconds'],pools=pools,coverage=result['coverage'],counts=result['counts'],
        unresolved=dict(collections.Counter(str(r.get('reason','unknown')) for r in result['rejected']))))


def report(export=False):
    bind();summary=BASE_REPORT(False)
    distribution=read(OUT/'face_size_distribution.json')
    if export:
        summary['final']=summary['pending_pairs']==0
        manifests.export(OUT,summary)
        write(OUT/'screening_summary.json',summary)
        distribution['final']=summary['final'];write(OUT/'face_size_distribution.json',distribution)
    write(OUT/'crop_size_distribution.json',dict(utc=summary['utc'],revision=crops.REVISION,
        complete_pairs=summary['complete_pairs'],total_pairs=summary['total_pairs'],final=summary['final'],
        pixel_edges=[None if x==float('inf') else x for x in crops.EDGES],area_fraction_edges=crops.AREA_EDGES,
        pools={k:dict(clips=v['clips'],seconds=v['seconds'],roles=v.get('crop_roles',{})) for k,v in distribution['pools'].items()},
        definitions=dict(side='Native-pixel fixed square head/shoulder proposal side; no upsampling floor or admission gate.',
            output_face_height='Median measured face height if the proposed square were resized to 512 pixels; statistics only.',
            face_area_in_crop='Median facial-landmark width times height divided by proposed crop area.',
            unmeasured='Still included in audio decisions, never treated as a zero-size valid crop.',
            head_fit='Landmark-based hair/forehead estimate, not verified full-head segmentation.',
            terminal_pools='Superseded parent candidates excluded; accepted children counted once.'),visual_filter_applied=False))
    return summary


def worker(node_name,local_rank):
    verify();bind()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    node=cluster.node_config(node_name);rank=cluster.worker_rank(node['index'],local_rank)
    lookup={r['pair']:r for r in read(OUT/'catalog.json')['rows']};queue=read(SCHED/'queue.json')['pairs']
    feedback=dict(rejects=read(OUT/'human_rejections.json')['rows'],positives=read(OUT/'human_retained.json')['rows'])
    counts=collections.Counter();timing=collections.Counter();models=[];started=time.monotonic();consecutive=0
    def progress(stage,pending_preparation=0,**extra):
        write(SCHED/'progress'/f'rank_{rank:03d}.json',dict(utc=full.now(),unix=time.time(),node=node_name,rank=rank,
            local_rank=local_rank,gpu=local_rank%8,pid=os.getpid(),stage=stage,pending_preparation=pending_preparation,
            counts=dict(counts),timing_seconds=dict(timing),wall_seconds=time.monotonic()-started,**extra))
    progress('starting')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={}
        def fill():
            ctrl=cluster.read(SCHED/'control.json');setting=ctrl['nodes'][node_name]
            active=rank_enabled(node,local_rank,setting) and not ctrl.get('stop') and not ctrl.get('drain')
            while active and len(pending)<setting['prep_per_worker']:
                pair=cluster.claim(queue,rank)
                if pair is None:break
                pending[pool.submit(prepare,lookup[pair])]=pair;counts['claimed_pairs']+=1
        while True:
            fill()
            if not pending:
                if cluster.exhausted() or cluster.read(SCHED/'control.json').get('stop'):break
                progress('idle_disabled');time.sleep(2);continue
            done,_=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:progress('preparing',len(pending));continue
            for future in done:
                pair=pending.pop(future);fill()
                try:
                    prepared=future.result();timing['preparation_sum']+=prepared['elapsed']
                    result,audit=screen_prepared(lookup[pair],prepared,models,feedback,
                        lambda **kw:progress(pending_preparation=len(pending),**kw))
                    result['scheduler']=dict(node=node_name,rank=rank)
                    write(OUT/'audit'/(pair+'.json'),audit)
                    write(OUT/'results'/(pair+'.json'),result);write_pair_summary(result)
                    counts.update(result['counts']);counts['complete_pairs']+=1;counts['clips']+=len(result['clips'])
                    timing['quality_and_statistics']+=result['timing_seconds']['quality_and_statistics'];consecutive=0
                except Exception as e:
                    counts['error_pairs']+=1;consecutive+=1
                    write(OUT/'errors'/(pair+'.json'),dict(pair=pair,utc=full.now(),error=repr(e),traceback=traceback.format_exc(),node=node_name,rank=rank))
                    if consecutive>=10:raise RuntimeError('Ten consecutive processing errors') from e
                progress('pair_finished',len(pending),pair=pair)
    progress('finished')


def supervise(attach=False):
    verify();bind();governor=governor_module.Governor()
    expected=[n['name'] for n in cluster.configuration()['nodes']]
    snapshot=governor_module.ResilientSnapshot(expected);began=time.time();reporter=None;last_report=0
    with (SCHED/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not attach:
                for n in cluster.configuration()['nodes']:
                    print(cluster.ssh(n,[cluster.configuration()['python'],str(SCRIPT),'detach-node','--node',n['name']]),flush=True)
            while True:
                snap=snapshot.snapshot();now=time.time();nodes=snap['nodes']
                complete=governor_module.validate_heartbeats(nodes,expected,now,began)
                if complete and all(n['state']=='finished' for n in nodes.values()):break
                ctrl=cluster.read(SCHED/'control.json');changed=False
                for name,node in nodes.items():
                    if node['state']=='finished' or node.get('telemetry_stale'):continue
                    setting,reasons=governor.consider(name,node,ctrl['nodes'][name],snap['node_counts'].get(name,0),now)
                    if setting:
                        with (SCHED/'governor/events.jsonl').open('a') as f:f.write(json.dumps(dict(utc=full.now(),node=name,previous=ctrl['nodes'][name],setting=setting,reasons=reasons))+'\n')
                        ctrl['nodes'][name]=setting;changed=True
                if changed:write(SCHED/'control.json',ctrl)
                if reporter is not None and reporter.poll() not in (None,0):raise RuntimeError('Statistics reporter failed')
                if now-last_report>300 and (reporter is None or reporter.poll() is not None):
                    with (OUT/'logs/report.log').open('ab',buffering=0) as f:
                        reporter=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'report'],stdin=subprocess.DEVNULL,stdout=f,stderr=f)
                    last_report=now
                write(SCHED/'latest.json',snap)
                write(SCHED/'governor/trials.json',dict(completed=governor.completed_trials,in_progress=governor.evaluations))
                write(OUT/'runtime.json',dict(state='cluster_screening',utc=full.now(),coordinator_pid=os.getpid(),
                    progress=snap['counts'],queue=snap['queue'],nodes={n:v['state'] for n,v in nodes.items()},
                    control=ctrl,visual_filter_applied=False,training_started=False))
                time.sleep(10)
            if reporter:reporter.wait()
            for p in (OUT/'results').glob('*.json'):
                if not (OUT/'summaries'/p.name).exists():write_pair_summary(read(p))
            summary=report(False)
            if summary['pending_pairs'] and cluster.read(SCHED/'control.json').get('stop'):
                write(OUT/'runtime.json',dict(state='stopped',utc=full.now(),summary=summary,workers=0,training_started=False));return
            assert summary['pending_pairs']==0,'Unaccounted claims; resume required'
            summary=report(True)
            write(OUT/'runtime.json',dict(state='finished_with_errors' if summary['error_pairs'] else 'finished',utc=full.now(),summary=summary,workers=0,training_started=False))
        except BaseException as e:
            ctrl=cluster.read(SCHED/'control.json');ctrl['stop']=True;write(SCHED/'control.json',ctrl)
            write(OUT/'runtime.json',dict(state='stopping_after_error',utc=full.now(),error=repr(e),training_started=False));raise


def detach(attach=False):
    verify();bind()
    with (SCHED/'logs/coordinator.log').open('ab',buffering=0) as f:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'supervise-existing' if attach else 'supervise'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
    write(SCHED/'launch.json',dict(utc=full.now(),pid=p.pid));print(json.dumps(read(SCHED/'launch.json')),flush=True)


def resume():
    verify();bind()
    code=health.process_probe(SCRIPT)
    for n in cluster.configuration()['nodes']:
        active=json.loads(cluster.ssh(n,[cluster.configuration()['python'],'-c',code]))
        if active:raise RuntimeError('Run-owned processes still active on '+n['name']+': '+str(active))
    history=SCHED/'attempts'/str(time.time_ns());history.mkdir(parents=True)
    for name in ('nodes','progress'):(SCHED/name).rename(history/name);(SCHED/name).mkdir()
    write(SCHED/'queue_state.json',dict(cursor=0,total=len(read(SCHED/'queue.json')['pairs'])))
    write(SCHED/'control.json',dict(stop=False,drain=False,stage='seamless_four_nodes',nodes=initial_nodes()))
    detach()


def smoke(pairs):
    verify();bind()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    catalog={r['pair']:r for r in read(OUT/'catalog.json')['rows']};models=[]
    feedback=dict(rejects=read(OUT/'human_rejections.json')['rows'],positives=read(OUT/'human_retained.json')['rows'])
    rows=[]
    for pair in pairs:
        t=time.monotonic();result,audit=screen_prepared(catalog[pair],prepare(catalog[pair]),models,feedback)
        target=OUT/'smoke';write(target/'results'/(pair+'.json'),result);write(target/'audit'/(pair+'.json'),audit)
        rows.append(dict(pair=pair,elapsed=time.monotonic()-t,terminal_clips=len(result['clips']),parents=len(result['superseded_attempts']),coverage=result['coverage'],counts=result['counts']))
        print(json.dumps(rows[-1]),flush=True)
    write(OUT/'smoke/summary.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','verify','smoke','report','detach','supervise','supervise-existing','detach-existing','worker','launch-node','detach-node','resume'])
    p.add_argument('--node');p.add_argument('--local-rank',type=int,default=0);p.add_argument('--export',action='store_true');p.add_argument('--pair',action='append')
    a=p.parse_args()
    if a.stage=='init':initialize()
    elif a.stage=='verify':verify();print('Verified')
    elif a.stage=='smoke':smoke(a.pair or ['naturalistic_V02_S3610_I00000088','improvised_V01_S0122_I00000125'])
    elif a.stage=='report':print(json.dumps(report(a.export)),flush=True)
    elif a.stage=='detach':detach()
    elif a.stage=='detach-existing':detach(True)
    elif a.stage=='resume':resume()
    elif a.stage=='supervise':supervise()
    elif a.stage=='supervise-existing':supervise(True)
    else:
        bind()
        if a.stage=='worker':worker(a.node,a.local_rank)
        elif a.stage=='launch-node':cluster.launch_node(a.node)
        elif a.stage=='detach-node':cluster.detach_node(a.node)
