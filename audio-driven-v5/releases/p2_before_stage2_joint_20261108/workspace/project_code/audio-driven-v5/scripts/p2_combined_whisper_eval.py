"""Step8000 handoff: combined6000/8000 on fixed Seamless200 and ResponseNet200."""
import argparse,copy,fcntl,json,os,subprocess,sys,time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition
TRAIN=ROOT/'experiments/p2_j_whisper_event_feature_train31_20261105'
DEFAULT=ROOT/'experiments/p2_combined_whisper_eval_20261106'
PARENTS=[ROOT/'experiments/p2_seamless200_whisper_20261105',
         ROOT/'experiments/p2_responsenet_random200_twelveway_20261105/whisper']


def link(src,dst):
    if dst.is_symlink():
        if dst.resolve()!=src.resolve():raise ValueError('Wrong reused source: '+str(dst))
        return
    if dst.exists():raise FileExistsError(dst)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src,target_is_directory=src.is_dir())


def same_values(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same_values(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same_values(x,y) for x,y in zip(a,b))
    return a==b


def reuse_gt_features(work,parents):
    """Merge only the scoped immutable NPZ leaves; each new cache owns its locks."""
    count=0
    for parent in parents:
        source=Path(parent['metrics_gt_cache'])
        for kind in ['faces','fid','fvd']:
            for src in (source/kind).glob('*/*.npz'):
                dst=work/'gt_cache'/src.relative_to(source)
                if dst.exists():
                    if dst.resolve()!=src.resolve() and sha(dst)!=sha(src):raise ValueError('Conflicting GT feature identity')
                else:link(src,dst);count+=1
    return count


@contextmanager
def combined_runtime():
    """Route the unchanged historical renderer through strict combined loaders.

    This process-local adapter never edits a checkpoint-bound source file. The
    source dictionary is evaluated before patching to avoid recursive imports.
    """
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_event_feature_train31 import install
    install(read(TRAIN/'config.json'))
    from p2_whisper_event_feature.inference import load_model
    from p2_whisper_event_feature.checkpoint import checked
    from p2_whisper_event_feature.config import source_identity
    expected=source_identity()
    with patch('p2_whisper_av.inference.load_model',load_model), \
         patch('p2_whisper_av.checkpoint.checked',checked), \
         patch('p2_whisper_av.config.source_identity',lambda:dict(expected)):
        if source_identity()!=expected:raise ValueError('Combined source adapter changed identity')
        yield


def worker(a):
    with combined_runtime():
        from p2_testsets_eval.whisper_runtime import worker as execute
        execute(a.work,a.stage,a.rank,a.limit,a.arms)


def exported(work,step):
    import torch
    from p2_listen.cache import atomic_torch
    from p2_whisper_event_feature.checkpoint import checked
    from p2_whisper_event_feature.config import source_identity
    original=TRAIN/f'step_{step:06d}.pt'
    target=work/'checkpoints'/original.name;receipt=target.with_suffix('.export.json')
    marker=read(original.with_suffix('.complete.json'))
    if receipt.exists():
        report=read(receipt)
        if report['original_sha256']!=marker['sha256']:raise ValueError('Original checkpoint changed')
        value=checked(target)
    else:
        value=checked(original)
        if value['identity']['source']!=source_identity():raise ValueError('Training source differs')
        model_only={k:value[k] for k in ['schema','step','identity','model']}
        atomic_torch(target,model_only)
        checksum=sha(target);save(target.with_suffix('.complete.json'),dict(step=step,sha256=checksum))
        reread=checked(target)
        if reread['identity']!=value['identity'] or set(reread['model'])!=set(value['model']):raise ValueError('Export metadata differs')
        if any(not torch.equal(t,reread['model'][k]) for k,t in value['model'].items()):raise ValueError('Export altered weights')
        report=dict(original_checkpoint=str(original),original_sha256=marker['sha256'],inference_checkpoint=str(target),
                    inference_sha256=checksum,all_model_tensors_bitwise_equal=True,step=step)
        save(receipt,report)
    if value['step']!=step or value['identity']!=read(TRAIN/'run.json'):raise ValueError('Wrong combined checkpoint')
    return value['identity'],report


def prepare(work):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_event_feature_train31 import install
    install(read(TRAIN/'config.json'))
    import torch
    torch.set_num_threads(2)
    from p2_testsets_eval.runtime import ready
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        for f,h in p['execution_code'].items():
            if sha(f)!=h:raise ValueError('Queued evaluation source changed')
        return p
    work.mkdir(parents=True,exist_ok=True)
    parents=[read(x/'protocol.json') for x in PARENTS]
    if any(x['generation']!=parents[0]['generation'] or x['seed']!=parents[0]['seed'] for x in parents):raise ValueError('Historical sampling differs')
    if parents[0]['whisper_encoder']!=parents[1]['whisper_encoder'] or parents[0]['feature_code']!=parents[1]['feature_code']:raise ValueError('Historical Whisper caches differ')
    cluster=read(TRAIN/'cluster.json');save(work/'cluster.json',cluster)
    arms=[];identity=None
    for step in [6000,8000]:
        value,audit=exported(work,step)
        if identity is not None and value!=identity:raise ValueError('Checkpoint training identities differ')
        identity=value
        arms.append(dict(name=f'base_event_feature_whisper_{step}',kind='combined_whisper',step=step,
                         checkpoint=audit['inference_checkpoint'],checkpoint_sha256=audit['inference_sha256'],
                         original_checkpoint=audit['original_checkpoint'],original_sha256=audit['original_sha256'],planner=parents[0]['planner']))
    if read(Path(identity['experiment']['whisper_cache'])/'protocol.json')['encoder']!=parents[0]['whisper_encoder']:raise ValueError('Speaker encoder differs from training')
    datasets={};all_items=[];parts=[[] for _ in range(31)]
    for parent,old in zip(PARENTS,parents):
        items=[x for rank in range(old['workers']) for x in read(parent/'shards'/f'{rank:02d}.json')]
        if len(items)!=200 or len({x['key'] for x in items})!=200:raise ValueError('Historical200 membership differs')
        for dataset,spec in old['datasets'].items():
            if sha(Path(spec['prepared'])/'references.jsonl')!=spec['references_sha256']:raise ValueError('References changed')
            datasets[dataset]=spec
            link(parent/'conditions'/dataset,work/'conditions'/dataset)
            link(parent/'whisper'/dataset,work/'whisper'/dataset)
        for item in items:
            folder=parent/'conditions'/item['key'];feature=parent/'whisper'/item['key']
            if not all(ready(x) for x in [folder/'qwen',folder/'text',feature]):raise ValueError('Incomplete historical condition')
            meta=read(feature/'metadata.json');plan=read(folder/'qwen/plan.json')
            if (meta['encoder']!=old['whisper_encoder'] or meta['feature_code']!=old['feature_code']
                or meta['source']['input_audit_sha256']!=sha(folder/'input_audit.json') or plan['adapter'] is not None):raise ValueError('Stale or non-base condition')
        for rank,shard in enumerate(partition(items,31)):parts[rank]+=shard
        all_items+=items
    neg=torch.load(PARENTS[0]/'negative/00/payload.pt',weights_only=True,map_location='cpu')
    other=torch.load(PARENTS[1]/'negative/00/payload.pt',weights_only=True,map_location='cpu')
    if not same_values(neg,other):raise ValueError('Dataset negative conditions differ')
    for rank in range(31):
        link(PARENTS[0]/'negative/00',work/'negative'/f'{rank:02d}')
        save(work/'shards'/f'{rank:02d}.json',parts[rank])
    save(work/'shared_items.json',all_items)
    p=copy.deepcopy(parents[0]);p.update(schema='combined-whisper-dual200-v1',arms=arms,datasets=datasets,
        workers=31,total_samples=400,total_outputs=800,checkpoint_identity=identity,
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),
        worker_cli=str(Path(__file__).resolve()),shared_item_queue=True,
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        metrics_gt_cache=str(ROOT/'experiments/p2_metrics_acceleration_20261102/gt_cache'),
        metrics_workers_per_gpu=4,metrics_use_released_inference_nodes=False,
        conditioning='historical base Qwen H/Gemma plus exact speaker-only Whisper; combined trained model',
        authorized_scope='31 released training GPUs; node8 GPU7 reserved; no eviction',
        comparison='fresh batch31 combined6000/8000; historical batch differences explicitly retained')
    p.pop('metrics_manifest_overrides',None)
    # Preserve the historical per-dataset GT features rather than mixing caches.
    p['metrics_gt_cache']=str(work/'gt_cache')
    cached_features=reuse_gt_features(work,parents)
    files=[Path(__file__).resolve(),ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/runtime.py',
           ROOT/'src/p2_testsets_eval/shared_items.py',ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_testsets_eval/metrics.py']
    p['execution_code']={str(f):sha(f) for f in files};save(work/'protocol.json',p)
    save(work/'CPU_PRECHECK.json',dict(success=True,samples=400,outputs=800,workers=31,checkpoints=[6000,8000],
        datasets=datasets,seed=p['seed'],generation=p['generation'],reused_source_conditions=400,
        reused_gt_feature_files=cached_features,reserved_gpu='node8:7',trained_model_gpu_smoke='pending',historical_results_untouched=True))
    return p


def smoke(work,p):
    """Eight real outputs across two datasets/checkpoints, retained for full run."""
    from p2_testsets_eval.cluster import run_cluster
    from p2_testsets_eval.runtime import ready,probe
    if (work/'GPU_SMOKE_COMPLETE.json').exists():return
    items=read(work/'shared_items.json');chosen=[]
    for dataset in p['datasets']:
        subset=sorted([x for x in items if x['dataset']==dataset],key=lambda x:x['geometry']['duration_s'])
        chosen.extend([subset[0],subset[-1]])
    dest=work/'gpu_smoke';dest.mkdir(exist_ok=True)
    cp=copy.deepcopy(p);cluster=read(p['cluster']);cluster['nodes']=cluster['nodes'][:1]
    cluster['nodes'][0]['gpus']=[0,1,2,3];save(dest/'cluster.json',cluster)
    cp.update(workers=4,total_samples=4,total_outputs=8,shared_item_queue=False,cluster=str(dest/'cluster.json'))
    save(dest/'protocol.json',cp)
    for folder in ['conditions','whisper','negative','latents','generated']:
        (work/folder).mkdir(exist_ok=True);link(work/folder,dest/folder)
    for rank,item in enumerate(chosen):save(dest/'shards'/f'{rank:02d}.json',[item])
    for stage in ['render','decode']:
        if not (dest/'stages'/(stage+'.json')).exists():run_cluster(dest,stage)
    for arm in p['arms']:
        for item in chosen:
            folder=work/'generated'/arm['name']/item['key']
            if not ready(folder):raise ValueError('Real combined GPU smoke incomplete')
            metadata=read(folder/'metadata.json')
            if metadata['step']!=arm['step'] or metadata['checkpoint_sha256']!=arm['original_sha256']:raise ValueError('Wrong smoke checkpoint')
            if metadata['target_description_access'] or metadata['listener_future_input']:raise ValueError('GT leakage')
            if probe(folder/'generated.mp4')['frames']!=item['geometry']['frames']:raise ValueError('Smoke time mismatch')
    save(work/'GPU_SMOKE_COMPLETE.json',dict(success=True,outputs=8,checkpoints=[6000,8000],
        items=[x['key'] for x in chosen],reused_by_full_run=True,at=time.time()))


def metrics_gate(work):
    """Native scorer gate plus two real smoke outputs per dataset.

    Native prepare_smoke takes the first two entries of two shards; combined
    shards contain both datasets, so use the explicitly validated GPU cases.
    """
    from evaluation.listener_metrics.suite import enrich
    p=read(work/'protocol.json');proof=read(work/'GPU_SMOKE_COMPLETE.json')
    if proof.get('success') is not True or proof['outputs']!=8:raise ValueError('Real GPU generation audit required')
    items={x['key']:x for x in read(work/'shared_items.json')};arm=p['arms'][0]['name']
    for dataset in p['datasets']:
        selected=[items[key] for key in proof['items'] if items[key]['dataset']==dataset]
        if len(selected)!=2:raise ValueError('Need two audited outputs per dataset')
        pairs=[];maps=[]
        for x in selected:
            r=x['reference'];folder=work/'generated'/arm/x['key']
            row=dict(id=r['id'],dataset=dataset,identity='listener',generated=str(folder/'generated.mp4'),listening_intervals=r['listening_intervals'])
            if dataset=='seamless':row['gt_reference']=r['gt_reference']
            else:row.update(gt=r['gt'],gt_resize=[512,512],gt_resize_method='pil_bicubic',source_start=r['source_start'],source_end=r['source_end'])
            pairs.append(row);maps.append(dict(id=r['id'],generated_audio=str(folder/'listener.wav')))
        pairs=enrich(pairs,[x['reference'] for x in selected],maps,work)
        path=work/'smoke_metric_manifests'/arm/dataset/'00.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(x)+'\n' for x in pairs))
    save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=8,derived_from='GPU_SMOKE_COMPLETE.json',at=time.time()))


def queue(work):
    from p2_testsets_eval.cluster import state,run_cluster,bind
    from p2_seamless200_whisper_eval import idle_snapshot
    work.mkdir(parents=True,exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        save(work/'CONTROLLER.json',process_record())
        while not (TRAIN/'STOP_AT_8000_COMPLETE.json').exists():
            if (work/'STOP').exists():state(work,'stopped');return
            state(work,'waiting_complete8000_and_training_stop',training=read(TRAIN/'status.json'));time.sleep(15)
        if read(TRAIN/'status.json')['step']!=8000:raise ValueError('Training did not stop exactly at8000')
        state(work,'preparing_dual200');p=prepare(work)
        for stage in ['gpu_smoke','render','decode']:
            if stage=='gpu_smoke' and (work/'GPU_SMOKE_COMPLETE.json').exists():continue
            if (work/'stages'/(stage+'.json')).exists():continue
            quiet=0
            while quiet<2:
                if (work/'STOP').exists():state(work,'stopped');return
                snap=idle_snapshot(read(p['cluster']));save(work/'resources.json',snap)
                quiet=quiet+1 if snap['all_idle'] else 0
                if quiet<2:state(work,'waiting_selected31_release',stage_pending=stage);time.sleep(15)
            for attempt in range(3):
                try:
                    if stage=='gpu_smoke':
                        state(work,'real_gpu_smoke');smoke(work,p)
                    else:run_cluster(work,stage)
                    break
                except Exception as exc:
                    state(work,'stage_retry',part=stage,attempt=attempt+1,error=repr(exc))
                    if attempt==2:raise
                    time.sleep(15)
        from p2_testsets_eval.runtime import ready
        items=read(work/'shared_items.json')
        if not all(ready(work/'generated'/a['name']/x['key']) for a in p['arms'] for x in items):raise ValueError('Missing committed generations')
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=800,at=time.time()))
        bind(work)
        metrics_gate(work)
        from p2_testsets_eval.metrics import queue as score
        score(work)
        historical={}
        for source in [ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105',PARENTS[0],PARENTS[1].parent]:
            historical.update(read(source/'RESULTS.json'))
        current=read(work/'RESULTS.json')
        save(work/'COMPARISON.json',dict(historical=historical,current=current,
            notes=['Identical fixed200 references and sampling; historical scores reused without regeneration.',
                   'Combined training uses batch31; historical training batches and durations differ.']))
        save(work/'HANDOFF_COMPLETE.json',dict(results=str(work/'RESULTS.json'),at=time.time(),training_resumed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['queue','prepare','worker'])
    p.add_argument('--work',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['cache','render','decode'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--arms',nargs='+');a=p.parse_args()
    if a.action=='worker':worker(a)
    elif a.action=='prepare':prepare(a.work)
    else:
        for attempt in range(3):
            try:queue(a.work);break
            except Exception as exc:
                save(a.work/'FAILURE.json',dict(at=time.time(),attempt=attempt+1,error=repr(exc)))
                save(a.work/'status.json',dict(stage='failed' if attempt==2 else 'retry_pending',attempt=attempt+1,error=repr(exc),at=time.time()))
                if attempt==2:raise
                time.sleep(30)
