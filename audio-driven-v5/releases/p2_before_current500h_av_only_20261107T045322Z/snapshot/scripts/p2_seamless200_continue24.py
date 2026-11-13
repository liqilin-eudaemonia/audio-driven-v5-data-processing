"""Move the paused benchmark to three nodes; reuse complete artifacts and add combined-loss7000."""
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition
from p2_response200_fourway import link

OLD=ROOT/'experiments/p2_seamless200_nineway_20261105'
TRAIN=ROOT/'experiments/p2_j_base_h_av_event_feature_continue24_20261105'


def prepare(work):
    import torch
    torch.set_num_threads(2)
    from p2_testsets_eval.runtime import ready,data_config,checkpoint_source
    previous=read(OLD/'protocol.json');refs=rows(Path(previous['datasets']['seamless']['prepared'])/'references.jsonl')
    if len(refs)!=200:raise ValueError('Expected identical frozen200')
    if read(TRAIN/'queue/state.json')['phase']!='stopped':raise ValueError('Training must have stopped')
    if not (OLD/'pause_for_other_job/CANCEL_AUTO_RESUME').exists():raise ValueError('Old node8 waiter must be disarmed')
    work.mkdir(exist_ok=False)
    cluster=read(TRAIN/'cluster.json')
    if [n['name'] for n in cluster['nodes']]!=['node17','node12','node4']:raise ValueError('Wrong selected hosts')
    save(work/'cluster.json',cluster);save(work/'metrics_cluster.json',cluster)
    p=copy.deepcopy(previous)
    p.update(created_at=time.time(),schema='seamless200-nine-plus-combined24-v1',workers=24,arms=[],total_outputs=2000,
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'),
        metrics_gt_cache=str(work/'gt_cache'),authorized_scope='Historical9 plus combined AV/Event/Feature7000; three nodes24GPU',
        execution_order='Reuse800 complete sampling outputs; decode missing only; native24GPU generation and GPU metrics',
        supersedes=str(OLD))
    items=[dict(key='seamless/'+r['id'],id=r['id'],dataset='seamless',reference=r,geometry=geometry(r,'seamless')) for r in refs]
    shards=partition(items,24)
    for i,s in enumerate(shards):save(work/'shards'/f'{i:02d}.json',s)
    checkpoint=TRAIN/'step_007000.pt';marker=read(checkpoint.with_suffix('.complete.json'))
    if marker['step']!=7000 or sha(checkpoint)!=marker['sha256']:raise ValueError('Combined7000 incomplete')
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    data_config();expected=checkpoint_source(saved['identity']['experiment'],'av_event_feature')
    migration=read(TRAIN/'MIGRATION.json');original=dict(saved['identity']['source'])
    for file,version in migration['source_changes'].items():
        if sha(file)!=version or original.pop(file,None)!=version:raise ValueError('Unapproved migration entry changed')
    if original!=expected:raise ValueError('Combined model implementation differs')
    feature_checkpoint=Path(read(OLD/'base_feature/protocol.json')['arms'][0]['checkpoint'])
    feature=torch.load(feature_checkpoint,map_location='cpu',mmap=True,weights_only=True)
    if saved['model'].keys()!=feature['model'].keys() or any(saved['model'][k].shape!=feature['model'][k].shape for k in saved['model']):
        raise ValueError('Combined/Feature architectures cannot share resident renderer')
    report=dict(checkpoint_source=saved['identity']['source'],current_source=expected,
        checkpoint_sha256=marker['sha256'],inference_math_unchanged=True,
        reason='Only extra source is the explicitly recorded training-only 28-to24 sampler/launcher; all inference source files identical',
        migration=str(TRAIN/'MIGRATION.json'),migration_sha256=sha(TRAIN/'MIGRATION.json'))
    compat=work/'compatibility/base_event_feature_7000.json';save(compat,report)
    combined=dict(name='base_event_feature_7000',kind='av_event_feature',step=7000,checkpoint=str(checkpoint),
        planner='unmodified_Qwen2.5-Omni-3B',source_compatibility=str(compat),
        samples_seen=190000,comparison_target_samples=192000,sample_count_difference=-2000)
    del saved,feature
    groups=[];reused=[]
    for name in ('base_standard','base_feature','sft_event'):
        source=OLD/name;child=work/name;child.mkdir();cp=copy.deepcopy(read(source/'protocol.json'))
        cp.update(workers=24,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'metrics_cluster.json'),
            metrics_gt_cache=str(work/'gt_cache'),execution_order=p['execution_order'])
        if name=='base_feature':cp['arms'].append(combined)
        cp['total_outputs']=200*len(cp['arms'])
        save(child/'protocol.json',cp)
        for i,s in enumerate(shards):save(child/'shards'/f'{i:02d}.json',s)
        for arm in cp['arms']:p['arms'].append(arm)
        conditions=source/'conditions'
        if conditions.is_dir():link(conditions,child/'conditions')
        else:(child/'conditions').mkdir()
        (child/'negative').mkdir()
        old_negative=source/'negative'
        if all(ready(old_negative/f'{i:02d}') for i in range(8)):
            # The native empty prompt is identical across ranks; check payload
            # bytes before sharing it with newly numbered ranks.
            hashes={sha(old_negative/f'{i:02d}'/'payload.pt') for i in range(8)}
            if len(hashes)!=1:raise ValueError('Native negative tensors need independent equality verification')
            for i in range(24):link(old_negative/f'{i%8:02d}',child/'negative'/f'{i:02d}')
        cache_complete=all(ready(child/'conditions'/x['key']/part) for x in items for part in ('qwen','text'))
        negatives_complete=all(ready(child/'negative'/f'{i:02d}') for i in range(24))
        if cache_complete and negatives_complete:
            save(child/'stages/cache.json',dict(complete=True,reused=True,samples=200,at=time.time()))
        all_latents=True
        for arm in cp['arms']:
            has=[]
            for item in items:
                old_latent=source/'latents'/arm['name']/item['key']
                if ready(old_latent):
                    metadata=read(old_latent/'metadata.json')
                    if (metadata['checkpoint']!=arm['checkpoint'] or metadata['seed']!=cp['seed'] or
                        metadata['generation']!=cp['generation'] or metadata['geometry']!=item['geometry']):
                        raise ValueError('Old sampling protocol differs')
                    link(old_latent,child/'latents'/arm['name']/item['key']);has.append(item['id'])
                    reused.append(dict(kind='latent',arm=arm['name'],id=item['id']))
                old_video=source/'generated'/arm['name']/item['key']
                if ready(old_video):
                    link(old_video,child/'generated'/arm['name']/item['key'])
                    reused.append(dict(kind='video',arm=arm['name'],id=item['id']))
            if len(has)!=200:all_latents=False
        if all_latents:save(child/'stages/render.json',dict(complete=True,reused=True,samples=200*len(cp['arms']),at=time.time()))
        groups.append(str(child))
    if len(p['arms'])!=10 or len({a['name'] for a in p['arms']})!=10:raise ValueError('Expected10 systems')
    save(work/'groups.json',groups);save(work/'protocol.json',p)
    save(work/'REUSE_AUDIT.json',dict(latents=sum(r['kind']=='latent' for r in reused),videos=sum(r['kind']=='video' for r in reused),
        records=reused,old_artifacts_not_overwritten=True,same_frozen200=True,other_node8_tasks_not_touched=True))
    save(work/'PRECHECK.json',dict(success=True,systems=10,samples=200,total_outputs=2000,workers=24,
        combined_checkpoint_sha256=marker['sha256'],combined_inference_architecture_matches_feature=True,
        combined_training_source_extra_only=True,training_stopped_step=read(TRAIN/'status.json')['step'],
        shard_counts=[len(s) for s in shards],same_generation_seed_steps_cfg_and_native_math=True))
    print('Prepared24GPU continuation; reused existing results',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    prepare(p.parse_args().work.resolve())
