"""Nine historical systems, one frozen latest-caption Seamless200, node8 eight GPUs."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition,PYTHON,METRIC_PYTHON
from p2_response200_fourway import link
OLD=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
NEW=ROOT/'experiments/p2_response200_feature_sft_event_20261103'
DATA=ROOT/'experiments/seamless_quick200_latest_20261105/prepared_200'
HOST='172.29.104.237'


def compatibility(identity,current,checkpoint,proof):
    """Only the EXACT changes previously verified by a real old-checkpoint replay."""
    changed={k for k in set(identity)|set(current) if identity.get(k)!=current.get(k)}
    approved={k for k in set(proof['checkpoint_source'])|set(proof['current_source'])
              if proof['checkpoint_source'].get(k)!=proof['current_source'].get(k)}
    if (not proof['inference_math_unchanged'] or not changed or not changed<=approved or
        any(identity.get(k)!=proof['checkpoint_source'].get(k) or current.get(k)!=proof['current_source'].get(k)
            for k in changed)):
        raise ValueError('Unverified source change for '+str(checkpoint))
    return dict(checkpoint_source=identity,current_source=current,checkpoint_sha256=read(checkpoint.with_suffix('.complete.json'))['sha256'],
        inference_math_unchanged=True,prior_proof=str(NEW/'sft_on_base_event/source_compatibility.json'),
        prior_gpu_replay=str(NEW/'sft_on_base_event/LEGACY_REPLAY_VERIFIED.json'),approved_changed_files=sorted(changed))


def prepare(work):
    import torch
    torch.set_num_threads(2)
    from evaluation.seamless.bind import load_references
    from p2_testsets_eval.runtime import data_config,checkpoint_source
    refs=load_references(DATA)
    if len(refs)!=200:raise ValueError('Expected frozen200')
    if not read(DATA.parent/'PREPARED_AUDIT.json')['success']:raise ValueError('Dataset audit incomplete')
    work.mkdir(exist_ok=False)
    base=read(OLD/'protocol.json');p=copy.deepcopy(base)
    p.update(schema='seamless200-nine-historical-v1',created_at=time.time(),workers=8,total_samples=200,total_outputs=1800,
        datasets=dict(seamless=dict(prepared=str(DATA),references_sha256=sha(DATA/'references.jsonl'),count=200,face_padding=0)),
        arms=[],cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'),
        planner='per_group_original_or_matching_SFT',metrics_use_released_inference_nodes=False,
        metrics_workers_per_gpu=4,metrics_inference_workers_per_gpu=4,enforce_planner_identity=True,
        metrics_face_device='cuda:0',metrics_emotion_device='cuda:0',metrics_pixel_workers=4,metric_batch_size=64,
        metrics_gt_cache=str(work/'gt_cache'),execution_order='resident-model groups; all8 inference then all8 metrics',
        authorized_scope='User confirmed all historical nine groups; do not include ongoing or Whisper checkpoints',
        selection=str(DATA.parent/'selection.jsonl'),selection_sha256=sha(DATA.parent/'selection.jsonl'),
        metric_sampling='unchanged historical formulas; full200 FID/FVD from all shard features')
    for k in ('supersedes','scope_history','deferred_datasets','scope_changed_at','scope_reason'):p.pop(k,None)
    c=read(ROOT/'configs/p2_j_qwen_whisper_av_train32_cluster_20261104.json')
    node=copy.deepcopy(next(n for n in c['nodes'] if n['name']=='node8'))
    node.update(host=HOST,role='master',gpus=list(range(8)));c['nodes']=[node]
    save(work/'cluster.json',c);save(work/'metrics_cluster.json',c)
    items=[dict(key='seamless/'+r['id'],id=r['id'],dataset='seamless',reference=r,geometry=geometry(r,'seamless')) for r in refs]
    shards=partition(items,8)
    for i,part in enumerate(shards):save(work/'shards'/f'{i:02d}.json',part)
    configs=[('base_standard',read(OLD/'protocol.json'),read(OLD/'protocol.json')['arms']),
             ('base_feature',read(NEW/'base_feature/protocol.json'),read(NEW/'base_feature/protocol.json')['arms']),
             ('sft_event',read(NEW/'sft_event/protocol.json'),read(NEW/'sft_event/protocol.json')['arms']+
                 read(NEW/'sft_on_base_event/protocol.json')['arms'])]
    original_proof=read(NEW/'sft_on_base_event/source_compatibility.json')
    if not read(NEW/'sft_on_base_event/LEGACY_REPLAY_VERIFIED.json'):raise ValueError('Missing historical GPU replay evidence')
    verified={};groups=[];data_config()
    for name,historical,arms in configs:
        child=work/name;child.mkdir();cp=copy.deepcopy(p)
        cp.update(arms=copy.deepcopy(arms),total_outputs=200*len(arms),planner=historical['planner'],
            planner_adapter=historical.get('planner_adapter'))
        for key in ('planner_adapter_identity','sft_hidden_cache'):
            if key in historical:cp[key]=historical[key]
            else:cp.pop(key,None)
        for arm in cp['arms']:
            path=Path(arm['checkpoint']);marker=read(path.with_suffix('.complete.json'))
            if path not in verified:
                if sha(path)!=marker['sha256']:raise ValueError('Checkpoint SHA mismatch')
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                if saved['step']!=arm['step']:raise ValueError('Checkpoint step mismatch')
                expected=checkpoint_source(saved['identity']['experiment'],arm['kind'])
                old_source=saved['identity']['source']
                verified[path]=dict(sha256=marker['sha256'],step=saved['step'],source=old_source,current_source=expected)
                del saved
            record=verified[path]
            arm['planner']=cp['planner']
            if record['source']!=record['current_source']:
                report=compatibility(record['source'],record['current_source'],path,original_proof)
                compat_path=work/'compatibility'/(arm['name']+'.json');save(compat_path,report)
                arm['source_compatibility']=str(compat_path)
            else:arm.pop('source_compatibility',None)
            p['arms'].append(arm)
        save(child/'protocol.json',cp)
        for i,part in enumerate(shards):save(child/'shards'/f'{i:02d}.json',part)
        groups.append(str(child))
    if len(p['arms'])!=9 or len({a['name'] for a in p['arms']})!=9:raise ValueError('Nine systems required')
    for folder in ('conditions','negative'):
        (work/'base_standard'/folder).mkdir(exist_ok=True)
        link(work/'base_standard'/folder,work/'base_feature'/folder)
    save(work/'groups.json',groups);save(work/'protocol.json',p)
    save(work/'checkpoint_hashes.json',{str(k):dict(sha256=v['sha256'],step=v['step']) for k,v in verified.items()})
    save(work/'PRECHECK.json',dict(success=True,systems=9,unique_checkpoints=len(verified),samples=200,
        output_count=1800,shard_counts=[len(s) for s in shards],input='speaker audio plus final-crop listener first frame',
        gt_description_not_input=True,latest_labels_not_rewritten=True,other_training_nodes_not_used=True))


def run(work):
    from p2_testsets_eval.cluster import run_cluster,bind,environment
    from p2_whisper8000_then_event_feature import pid_capacity
    from p2_j_event_feature_launch import resources
    from listen_cluster import transport
    work=work.resolve();p=read(work/'protocol.json');c=read(work/'cluster.json')
    with (work/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        save(work/'controller.json',process_record())
        def state(stage,**kw):save(work/'status.json',dict(stage=stage,at=time.time(),**kw));print(stage,kw,flush=True)
        def publish(child):
            names=[a['name'] for a in read(child/'protocol.json')['arms']]
            for name in names:
                for folder in ('generated','maps','bindings','metric_manifests'):
                    link(child/folder/name,work/folder/name)
                save(work/'stages'/('bind_'+name+'.json'),dict(complete=True,at=time.time()))
            if not (work/'SMOKE_COMPLETE.json').exists():
                save(work/'SMOKE_COMPLETE.json',dict(status='production_generation_passed',samples=200,
                    arm=names[0],ranks=8,at=time.time()))
            return names
        try:
            groups=[Path(g) for g in read(work/'groups.json')]
            for child in groups:
                if (work/'STOP').exists():state('stopped');return
                if (child/'GENERATION_COMPLETE.json').exists():publish(child);continue
                while True:
                    checks=resources(c)+pid_capacity(c);save(work/'resources.json',dict(at=time.time(),nodes=checks))
                    if all(r['free'] for r in checks):break
                    if (work/'STOP').exists():state('stopped');return
                    state('waiting_node8_resources',group=child.name);time.sleep(10)
                state('generating',group=child.name)
                # Persistent stages load Qwen/Gemma/LTX/VAE once per GPU, share
                # base planner caches across six systems, and reuse model loads
                # across every compatible checkpoint in a group.
                for stage in ('cache','render','decode'):
                    if (work/'STOP').exists():state('stopped');return
                    if not (child/'stages'/(stage+'.json')).exists():run_cluster(child,stage)
                names=[a['name'] for a in read(child/'protocol.json')['arms']]
                if not (child/'stages/bind_all.json').exists():
                    bind(child,names);save(child/'stages/bind_all.json',dict(complete=True,at=time.time()))
                save(child/'GENERATION_COMPLETE.json',dict(total_outputs=200*len(names),at=time.time()))
                publish(child)
                state('group_generation_complete',group=child.name,arms=names)
            save(work/'GENERATION_COMPLETE.json',dict(total_outputs=1800,at=time.time()))
            state('metrics',systems=9,samples=200)
            # Existing dynamic metric scheduler: four workers per GPU, one host;
            # all 72 jobs are available and shared GT features are reused.
            command=['env','PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',
                'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',str(PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),
                'metrics_queue','--work',str(work)]
            with (work/'metrics_queue.log').open('a') as log:
                result=subprocess.run(transport(c,c['nodes'][0],command),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not (work/'COMPLETE.json').exists():raise RuntimeError('Metrics did not finish; see metrics_queue.log')
            state('complete' if read(work/'COMPLETE.json')['status']=='complete' else 'completed_with_metric_failures',
                results=str(work/'RESULTS.json'))
        except Exception as exc:state('failed',error=repr(exc));raise


def owned_workers_active(work):
    """Called on node8; never confuse a controller crash with its surviving workers."""
    import os
    for folder in [work,*[Path(g) for g in read(work/'groups.json')]]:
        for sub in ('processes','progress'):
            root=folder/sub
            if not root.exists():continue
            for path in root.rglob('*.json'):
                try:
                    record=read(path);pid=record.get('pid')
                    if type(pid) is not int:continue
                    p=Path('/proc')/str(pid)
                    args=(p/'cmdline').read_bytes().replace(b'\0',b' ')
                    if str(work).encode() in args and b'p2_testsets_eval.py' in args:
                        return True
                except (OSError,ValueError):continue
    # A metric scheduler can survive its outer controller; let it finish.
    p=work/'metrics_queue.lock'
    if p.exists():
        with p.open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
    return False


def watch(work):
    """Finite job guardian, ends with this evaluation; no recurring user reports."""
    from p2_speaker_whisper_after_annotation import alive_record,process_record
    work=work.resolve()
    with (work/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        restarts=0
        while not (work/'STOP').exists() and not (work/'COMPLETE.json').exists():
            r=read(work/'controller.json') if (work/'controller.json').exists() else {}
            alive=alive_record(r,Path(__file__).resolve(),'run')
            active=not alive and owned_workers_active(work)
            save(work/'watch.json',dict(**process_record(),controller_alive=alive,
                surviving_owned_workers=active,restarts=restarts))
            if not alive and not active:
                if restarts>=3:
                    save(work/'status.json',dict(stage='needs_attention',at=time.time(),reason='controller retry budget exhausted'));return
                with (work/'queue.log').open('a') as log:
                    subprocess.Popen([str(PYTHON),'-u',str(Path(__file__).resolve()),'run','--work',str(work)],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                restarts+=1
            time.sleep(15)
        save(work/'watch_finished.json',dict(at=time.time(),complete=(work/'COMPLETE.json').exists(),
            stopped=(work/'STOP').exists(),restarts=restarts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','watch'))
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    {'prepare':prepare,'run':run,'watch':watch}[a.command](a.work.resolve())
