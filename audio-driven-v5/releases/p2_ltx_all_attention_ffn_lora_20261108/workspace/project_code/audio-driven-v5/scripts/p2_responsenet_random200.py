"""Fixed stratified random ResponseNet200, twelve matched historical systems."""
import argparse
import copy
import csv
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition,PYTHON,METRIC_PYTHON
from p2_response200_fourway import link
OLD=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
PREVIOUS=ROOT/'experiments/p2_response200_feature_sft_event_20261103'
TEN=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
WHISPER=ROOT/'experiments/p2_seamless200_whisper_20261105'
DEFAULT=ROOT/'experiments/p2_responsenet_random200_twelveway_20261105'
SEED=20261105


def stratified_sample(refs,gt_audio,n=200,seed=SEED):
    """Hamilton quotas; randomized ties and within-stratum uniform draws.

    Uses only GT acoustic/nonzero state, clip duration and ASR provenance;
    never model outputs, visual quality scores or model rankings.
    """
    rng=random.Random(seed);strata=defaultdict(list)
    def kind(r):
        a=gt_audio[r['id']]['audio']['gt']
        sound='active' if a['acoustic'] else 'nonzero_below_activity' if a['peak']>0 else 'zero'
        duration='lt5' if r['duration_s']<5 else '5to8' if r['duration_s']<8 else 'ge8'
        return (sound,duration,r['group'])
    for r in sorted(refs,key=lambda r:r['id']):strata[kind(r)].append(r)
    if not 0<n<=len(refs):raise ValueError('Invalid random subset size')
    keys=sorted(strata);rng.shuffle(keys)
    quotas={k:math.floor(len(strata[k])*n/len(refs)) for k in keys}
    order=sorted(keys,key=lambda k:-(len(strata[k])*n/len(refs)-quotas[k]))
    for k in order[:n-sum(quotas.values())]:quotas[k]+=1
    chosen=[];stats=[]
    for key in sorted(strata):
        chosen+=rng.sample(strata[key],quotas[key])
        stats.append(dict(audio=key[0],duration=key[1],asr_source=key[2],population=len(strata[key]),
                          selected=quotas[key],ideal=len(strata[key])*n/len(refs)))
    rng.shuffle(chosen)
    if len(chosen)!=n or len({r['id'] for r in chosen})!=n:raise ValueError('Subset count/duplicate')
    return chosen,stats


def dump_rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))


def prepare(work):
    from p2_testsets_eval.runtime import ready
    if (work/'PRECHECK.json').exists():return read(work/'protocol.json')
    work.mkdir(parents=True,exist_ok=True)
    original=read(OLD/'protocol.json');refpath=Path(original['datasets']['responsenet']['prepared'])/'references.jsonl'
    refs=rows(refpath)
    if len(refs)!=586 or sha(refpath)!=original['datasets']['responsenet']['references_sha256']:
        raise ValueError('Original full586 references changed')
    gt_audio={r['id']:r for r in rows(OLD/'metrics/av_only_6000/responsenet/audio_pairs.jsonl')}
    chosen,strata=stratified_sample(refs,gt_audio)
    selected_ids={r['id'] for r in chosen}
    prepared=work/'references';prepared.mkdir(exist_ok=True)
    dump_rows(prepared/'references.jsonl',chosen)
    save(prepared/'seal.json',dict(references_sha256=sha(prepared/'references.jsonl'),
        parent=str(refpath),parent_sha256=sha(refpath),sample_size=200,seed=SEED))
    selection=[dict(id=r['id'],source_id=r['sample_id'],source_start=r['source_start'],source_end=r['source_end'],
        duration_s=r['duration_s'],asr_source=r['group'],audio='active' if gt_audio[r['id']]['audio']['gt']['acoustic']
        else 'nonzero_below_activity' if gt_audio[r['id']]['audio']['gt']['peak']>0 else 'zero',
        reason='stratified uniform random; no generated score or caption used',seed=SEED) for r in chosen]
    dump_rows(work/'selection.jsonl',selection)
    with (work/'selection.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
    save(work/'selection_summary.json',dict(population=586,selected=200,seed=SEED,
        strata=strata,source_videos=len({r['sample_id'] for r in chosen}),population_sources=len({r['sample_id'] for r in refs}),
        source_counts=dict(Counter(r['sample_id'] for r in chosen)),audio_counts=dict(Counter(r['audio'] for r in selection)),
        duration_s=sum(r['duration_s'] for r in chosen),uses_model_scores=False,
        overlap_old_selected200=len(selected_ids&{r['id'] for r in rows(ROOT/'experiments/responsenet_initial200_20261102/references.jsonl')})))
    cluster=read(WHISPER/'cluster.json');save(work/'cluster.json',cluster)
    p=copy.deepcopy(read(TEN/'protocol.json'))
    wp=read(WHISPER/'protocol.json')
    p.update(schema='responsenet-stratified-random200-twelveway-v1',created_at=time.time(),workers=28,
        datasets=dict(responsenet=dict(prepared=str(prepared),references_sha256=sha(prepared/'references.jsonl'),count=200,face_padding=64)),
        total_samples=200,total_outputs=2400,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),
        metrics_gt_cache=str(ROOT/'experiments/p2_metrics_acceleration_20261102/gt_cache'),
        metrics_use_released_inference_nodes=False,metrics_workers_per_gpu=4,metrics_inference_workers_per_gpu=4,
        authorized_scope='12 systems on fixed independently stratified random200; four hosts, authorized28 GPUs only; no other-task termination',
        selection=str(work/'selection.jsonl'),selection_sha256=sha(work/'selection.jsonl'),
        inference_nodes=[n['name'] for n in cluster['nodes']],arms=copy.deepcopy(read(TEN/'protocol.json')['arms'])+copy.deepcopy(wp['arms']))
    p.pop('metrics_decoded_gt_cache',None)
    p['execution_order']='reuse old output -> cache missing conditions -> resident compatible models -> native export -> reuse old scores -> all200 global feature merge'
    p['dependency']=str(WHISPER)
    if len(p['arms'])!=12 or len({a['name'] for a in p['arms']})!=12:raise ValueError('Expected twelve systems')
    items=[dict(key='responsenet/'+r['id'],id=r['id'],dataset='responsenet',reference=r,geometry=geometry(r,'responsenet')) for r in chosen]
    shards=partition(items,28)
    for rank,part in enumerate(shards):save(work/'shards'/f'{rank:02d}.json',part)
    templates=dict(base_standard=read(OLD/'protocol.json'),base_feature=read(PREVIOUS/'base_feature/protocol.json'),
        sft_event=read(PREVIOUS/'sft_event/protocol.json'),combined=read(OLD/'protocol.json'),whisper=wp)
    members=dict(base_standard=['av_only_6000','av_only_8927','av_event_6000','av_event_8927'],
        base_feature=['base_feature_6000','base_feature_8927'],
        sft_event=['sft_event_6000','sft_event_8927','sft_on_base_event_6000'],
        combined=['base_event_feature_7000'],whisper=['whisper_av_6000','whisper_av_8000'])
    reuse=[];groups=[];byname={a['name']:a for a in p['arms']}
    for name,names in members.items():
        child=work/name;child.mkdir(exist_ok=True);cp=copy.deepcopy(p);historical=templates[name]
        cp.update(arms=[byname[a] for a in names],total_outputs=200*len(names),planner=historical['planner'],
            planner_adapter=historical.get('planner_adapter'))
        for key in ['planner_adapter_identity','sft_hidden_cache']:
            if key in historical:cp[key]=historical[key]
            else:cp.pop(key,None)
        if name=='whisper':
            cp.update(worker_cli=str(Path(__file__).resolve()),whisper_encoder=wp['whisper_encoder'],
                whisper_batch=wp['whisper_batch'],feature_code=wp['feature_code'],checkpoint_identity=wp['checkpoint_identity'],sampling=wp['sampling'])
        # Full586 base Qwen/Gemma inputs already exist. All native-H groups share them.
        if name!='sft_event':link(OLD/'conditions',child/'conditions')
        else:
            for item in items:
                src=PREVIOUS/'sft_event/conditions'/item['key']
                if ready(src/'qwen') and ready(src/'text'):link(src,child/'conditions'/item['key'])
        for rank in range(28):
            src=OLD/'negative'/f'{rank:02d}' if name!='sft_event' else PREVIOUS/'sft_event/negative'/f'{rank:02d}'
            link(src,child/'negative'/f'{rank:02d}')
            save(child/'shards'/f'{rank:02d}.json',shards[rank])
        for arm in cp['arms']:
            a=arm['name']
            for item in items:
                sources=[OLD/'generated'/a/item['key'],PREVIOUS/'generated'/a/item['key']]
                src=next((s for s in sources if ready(s)),None)
                if src is None:continue
                metadata=read(src/'metadata.json');marker=read(Path(arm['checkpoint']).with_suffix('.complete.json'))
                if not (metadata['arm']==a and metadata['step']==arm['step'] and metadata['seed']==p['seed']
                        and metadata['generation']==p['generation'] and metadata['checkpoint_sha256']==marker['sha256']
                        and metadata['target_description_access'] is False and metadata['listener_future_input'] is False):
                    raise ValueError('Historical generation protocol mismatch: '+str(src))
                link(src,child/'generated'/a/item['key'])
                old_latents=src.parent.parent.parent.parent/'latents'/a/item['key']
                # Resolve actual per-pipeline output root before locating retained latents.
                old_latents=src.resolve().parents[3]/'latents'/a/item['key']
                if ready(old_latents):link(old_latents,child/'latents'/a/item['key'])
                reuse.append(dict(arm=a,id=item['id'],source=str(src.resolve()),sha256=sha(src/'metadata.json')))
            link(child/'generated'/a,work/'generated'/a)
        save(child/'protocol.json',cp);groups.append(str(child))
    for item in items:
        folder=OLD/'conditions'/item['key']
        if not ready(folder/'qwen') or not ready(folder/'text'):raise ValueError('Missing base Qwen/Gemma condition')
        plan=read(folder/'qwen/plan.json')
        if plan['adapter'] is not None or plan['target_description_access']:raise ValueError('Base input identity mismatch')
        if not read(folder/'input_audit.json')['source_only']:raise ValueError('Non-source-only input')
    code=[Path(__file__).resolve(),ROOT/'src/p2_testsets_eval/reuse_scores.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',
        ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_testsets_eval/runtime.py']
    p['execution_code']={str(f):sha(f) for f in code}
    for child in map(Path,groups):
        cp=read(child/'protocol.json');cp['execution_code']=p['execution_code'];save(child/'protocol.json',cp)
    save(work/'protocol.json',p);save(work/'groups.json',groups)
    save(work/'REUSED_GENERATIONS.json',dict(count=len(reuse),by_arm=dict(Counter(x['arm'] for x in reuse)),records=reuse))
    save(work/'PRECHECK.json',dict(success=True,samples=200,arms=12,total_outputs=2400,
        reused_outputs=len(reuse),new_outputs=2400-len(reuse),stratification='GT audio x duration x ASR source',
        model_scores_used=False,source_only=True,shards=[len(s) for s in shards],gpu_new_outputs_not_yet_run=True,at=time.time()))
    save(work/'status.json',dict(stage='prepared',at=time.time(),outputs=2400,reused=len(reuse)))
    return p


def rn_speaker_wave(item,folder):
    """Exact full pair speaker WAV, same mono/resampling/endpoint rule as base Qwen."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audit=read(folder/'input_audit.json')
    if not (audit['source_only'] and not audit['target_description_in_model_input'] and not audit['listener_future_media_in_model_input']):
        raise ValueError('Speaker-only input audit failed')
    source=Path(item['reference']['speaker']['audio'])
    if Path(audit['speaker_audio']).resolve()!=source.resolve():raise ValueError('Whisper source differs from Qwen speaker')
    wave,rate=sf.read(source,dtype='float32',always_2d=True)
    header=item['reference']['media']['speaker']['audio']
    if len(wave)!=header['samples'] or rate!=header['sample_rate']:raise ValueError('Speaker header changed')
    wave=wave.mean(1)
    if rate!=16000:
        g=math.gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g).astype(np.float32)
    count=round(item['geometry']['duration_s']*16000);delta=count-len(wave)
    if abs(delta)>16 or not np.isfinite(wave).all():raise ValueError('Speaker endpoint mismatch')
    wave=np.pad(wave,(0,max(0,delta)))[:count]
    return wave,dict(input_audit_sha256=sha(folder/'input_audit.json'),source_path=str(source),source_sha256=sha(source),
        waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),samples=count,rate=16000,
        origin_s=0.,silence_removed=False,source_only=True,endpoint_adjustment_samples=delta)


def run(work):
    from p2_testsets_eval.cluster import run_cluster,bind,state,stopped
    from p2_testsets_eval.runtime import ready
    from p2_seamless200_whisper_eval import idle_snapshot
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=prepare(work)
        for f,h in p['execution_code'].items():
            if sha(f)!=h:raise ValueError('Prepared code changed: '+f)
        try:
            dependency=Path(p['dependency'])
            while not (dependency/'PUBLISHED.json').exists():
                if stopped(work):state(work,'stopped');return
                state(work,'waiting_prior_whisper_supplement',dependency=str(dependency));time.sleep(30)
            groups=list(map(Path,read(work/'groups.json')))
            def idle():
                while True:
                    if stopped(work):state(work,'stopped');return False
                    snap=idle_snapshot(read(p['cluster']));save(work/'resources.json',snap)
                    if snap['all_idle']:return True
                    state(work,'waiting_authorized28_gpu_release',resources=snap);time.sleep(30)
            for child in groups:
                cp=read(child/'protocol.json');arms=cp['arms']
                if not all(ready(child/'generated'/a['name']/x['key']) for a in arms for k in range(p['workers']) for x in read(child/'shards'/f'{k:02d}.json')):
                    for stage in ['cache','render','decode']:
                        if (child/'stages'/(stage+'.json')).exists():continue
                        if not idle():return
                        state(work,'generating',group=child.name,part=stage)
                        run_cluster(child,stage)
                state(work,'binding',group=child.name)
                bind(child)
                for a in arms:
                    for folder in ['bindings','maps','metric_manifests']:link(child/folder/a['name'],work/folder/a['name'])
                    save(work/'stages'/('bind_'+a['name']+'.json'),dict(complete=True,at=time.time()))
                save(child/'GENERATION_COMPLETE.json',dict(total_outputs=cp['total_outputs'],at=time.time()))
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=2400,at=time.time()))
            # CPU repartition reuses all four existing full586 score sets on these
            # exact selected IDs, including exact FID/FVD feature rows.
            state(work,'reusing_original_full586_scores')
            command=[str(METRIC_PYTHON),str(Path(__file__).resolve()),'reuse_scores','--work',str(work)]
            from p2_testsets_eval.cluster import environment
            with (work/'reuse_scores.log').open('a') as log:
                subprocess.run(command,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
            save(work/'SMOKE_COMPLETE.json',dict(status='production_generation_passed',samples=200,arms=12,at=time.time()))
            state(work,'metrics')
            from p2_testsets_eval.metrics import queue
            queue(work)
            publish(work)
        except Exception as exc:state(work,'failed',error=repr(exc));raise


def publish(work):
    from p2_testsets_eval.cluster import state
    p=read(work/'protocol.json');result=read(work/'RESULTS.json');records=[]
    for arm in p['arms']:
        s=result[arm['name']+'/responsenet'];records.append(dict(arm=arm['name'],step=arm['step'],**s['metrics'],
            missed_response_rate=s['audio']['activity']['acoustic']['missed_response_rate'],
            false_sound_clip_rate=s['audio']['activity']['acoustic']['false_sound_clip_rate']))
    with (work/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    text=['# ResponseNet分层随机200条：12组结果','',
        '随机种子20261105；按GT音频状态、时长、ASR来源分层，未按模型得分选样。',
        '视频512×512，原30fps，seed137，30步Euler，CFG3/7。原Qwen／SFT Qwen按对应pipeline使用。','',
        '| 配置 | PSNR↑ | SSIM↑ | FID↓ | FVD↓ | LPIPS↓ | Emo-Acc↑ | ΔResp↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:text.append('| '+record['arm']+' | '+' | '.join(f'{record[k]:.5f}' if record[k] is not None else 'N/A' for k in ['psnr','ssim','fid','fvd','lpips','emo_acc','delta_resp'])+' |')
    text+=['','FID/FVD均由当前200条特征全局重算；原完整586条四组分数仅在ID／输入完全相同时重用逐条值，未使用完整586条的分布指标替代。',
        '短回应／静音的UTMOS按原规则保留N/A和有效覆盖率。单种子结果，细小差异不能视为稳定优势。']
    (work/'RESULTS.md').write_text('\n'.join(text)+'\n')
    preview=ROOT/'data_previews/listen_benchmark_review/p2_responsenet_random200_20261105'
    if not preview.exists():preview.symlink_to(work,target_is_directory=True)
    state(work,'completed_with_metric_failures' if read(work/'COMPLETE.json')['status']!='complete' else 'complete',results=str(work/'RESULTS.json'))
    save(work/'PUBLISHED.json',dict(results=str(work/'RESULTS.json'),at=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','queue','worker','reuse_scores','publish'])
    p.add_argument('--work',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['cache','render','decode'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--arms',nargs='+');a=p.parse_args()
    if a.action=='worker':
        import p2_testsets_eval.whisper_runtime as runtime
        runtime.speaker_wave=rn_speaker_wave
        runtime.worker(a.work,a.stage,a.rank,a.limit,a.arms)
    elif a.action=='reuse_scores':
        from p2_testsets_eval.reuse_scores import repartition
        repartition(a.work,OLD,[a['name'] for a in read(OLD/'protocol.json')['arms']])
    else:{'prepare':prepare,'queue':run,'publish':publish}[a.action](a.work)

if __name__=='__main__':main()
