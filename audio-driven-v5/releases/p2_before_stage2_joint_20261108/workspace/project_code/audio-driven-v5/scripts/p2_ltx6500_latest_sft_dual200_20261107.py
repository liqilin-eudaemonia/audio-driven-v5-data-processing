"""Same latest-SFT dual200 conditions; evaluate the24-card continuation6500 next."""
import argparse,copy,hashlib,inspect,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha
import p2_ltx6000_latest_sft_dual200_20261107 as base
WORK=ROOT/'experiments/p2_ltx6500_latest_sft_dual200_20261107'
UPSTREAM=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
TRAIN=ROOT/'experiments/p2_ltx_clip_train24_reserve_node11_20261107'
ARM='ltx_hidden_event_feature_whisper_6500_sft2752'


def install():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_ltx_events.config import source_identity,validate
    expected=read(TRAIN/'run.json')
    if source_identity()!=expected['source']:raise ValueError('6500 inference sources differ from native24-card checkpoint')
    validate(expected['experiment'])


def code_identity():
    # Keep the exact inherited generation implementation and record this adapter.
    return dict(base.code_identity(),**{str(SELF):sha(SELF),str(ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py'):sha(ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py')})


def prepare(work):
    work.mkdir(exist_ok=True)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        if p['execution_code']!=code_identity():raise ValueError('Queued6500 implementation changed')
        return p
    install()
    import torch
    torch.set_num_threads(2)
    from p2_ltx_events.checkpoint import checked
    from p2_ltx_events.config import source_identity,validate
    from p2_listen.cache import atomic_torch
    from p2_testsets_eval.runtime import ready
    from p2_combined_whisper_eval import link
    save(work/'status.json',dict(stage='cpu_preparing',at=time.time()))
    parent=read(UPSTREAM/'protocol.json');items=read(UPSTREAM/'shared_items.json')
    assert len(items)==400 and sha(UPSTREAM/'shared_items.json')==parent['shared_items_sha256']
    original=TRAIN/'step_006500.pt';marker=read(original.with_suffix('.complete.json'))
    target=work/'checkpoints/step_006500.pt';receipt=target.with_suffix('.export.json')
    if receipt.exists():
        audit=read(receipt);assert audit['original_sha256']==marker['sha256'];saved=checked(target)
    else:
        saved=checked(original)
        assert saved['step']==6500 and saved['identity']==read(TRAIN/'run.json') and saved['identity']['source']==source_identity()
        atomic_torch(target,{k:saved[k] for k in ['schema','step','identity','model']})
        save(target.with_suffix('.complete.json'),dict(step=6500,sha256=sha(target)))
        exported=checked(target)
        assert exported['identity']==saved['identity'] and set(exported['model'])==set(saved['model'])
        assert all(torch.equal(v,exported['model'][k]) for k,v in saved['model'].items())
        audit=dict(step=6500,original_checkpoint=str(original),original_sha256=marker['sha256'],inference_checkpoint=str(target),inference_sha256=read(target.with_suffix('.complete.json'))['sha256'],model_bitwise_equal=True,optimizer_omitted=True)
        save(receipt,audit)
    identity=saved['identity'];c=validate(identity['experiment']);old=parent['checkpoint_identity']['experiment']
    for k in ['loss','loss_weights','hidden_event_supervision','current_cache_index','qwen_hidden_source','connector','whisper']:
        if c[k]!=old[k]:raise ValueError('6500 changed model/data specification: '+k)
    cached=[]
    for dataset in parent['datasets']:
        link(UPSTREAM/'conditions'/dataset,work/'conditions'/dataset)
        link(UPSTREAM/'whisper'/dataset,work/'whisper'/dataset)
    for x in items:
        folder=work/'conditions'/x['key'];assert ready(folder/'qwen') and ready(folder/'text')
        if read(folder/'qwen/plan.json')['adapter']!=parent['planner_adapter_identity']:raise ValueError('Not the same latest SFT2752 condition')
        cached.append(x['key'])
    for rank in range(8):
        link(UPSTREAM/'negative'/f'{rank:02d}',work/'negative'/f'{rank:02d}')
        save(work/'shards'/f'{rank:02d}.json',read(UPSTREAM/'shards'/f'{rank:02d}.json'))
    save(work/'shared_items.json',items)
    cluster=read(UPSTREAM/'cluster.json');cluster['training_script']=str(SELF);save(work/'cluster.json',cluster)
    p=copy.deepcopy(parent);p.update(schema='ltx-hidden-event6500-latest-sft2752-dual200-v1',created_at=time.time(),
        arms=[dict(name=ARM,kind='ltx_hidden_event_feature_whisper',step=6500,checkpoint=str(target),checkpoint_sha256=audit['inference_sha256'],original_checkpoint=str(original),original_sha256=marker['sha256'],planner=parent['planner'])],
        checkpoint_identity=identity,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(SELF),
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        upstream_completion=str(UPSTREAM/'HANDOFF_COMPLETE.json'),upstream_work=str(UPSTREAM),
        authorized_scope='node11 GPUs0..7 after6000+SFT2752 completes; same frozen conditions; no other-task termination',
        comparison='Same400 SFT2752/H/Gemma/Whisper conditions as6000; training continued batch31 to6026 thenbatch24 to6500',
        execution_order='wait6000 scores complete -> reused400 conditions -> shared8 render -> decode -> full200 native metrics',execution_code=code_identity())
    save(work/'protocol.json',p)
    save(work/'stages/cache.json',dict(complete=True,reused_from=str(UPSTREAM),samples=400,latest_sft_step=2752,at=time.time()))
    save(work/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),ltx_step=6500,sft_step=2752,model_export=audit,conditions_reused_exactly=400,source_identity_exact=True,identical_test_refs=parent['datasets'],identical_sampling=p['generation'],seed=p['seed'],gpu_validation='pending after6000 finishes'))
    return p


def final(work,p):
    from p2_testsets_eval.runtime import ready
    results=read(work/'RESULTS.json');items=read(work/'shared_items.json')
    for x in items:
        folder=work/'generated'/ARM/x['key'];assert ready(folder)
        m=read(folder/'metadata.json');assert m['step']==6500 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
        assert m['planning_adapter']==p['planner_adapter_identity'] and not m['target_description_access'] and not m['listener_future_input']
    for d in p['datasets']:
        s=results[ARM+'/'+d];assert s['requested_pairs']==200 and s['audio']['requested_pairs']==200 and s['audio']['pairs_with_errors']==0
        assert all(s['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),ltx_step=6500,sft_step=2752,outputs=400,identical6000_conditions=True))
    save(work/'COMPARISON.json',dict(current=results,step6000=read(UPSTREAM/'RESULTS.json'),notes=[p['comparison']]))
    (work/'RESULTS.md').write_text('# LTX主干整段Event＋Feature＋Whisper6500＋最新SFT2752\n\n与6000步使用同一400条条件、测试清单、采样和指标；原值及覆盖见RESULTS.json，6000/6500对照见COMPARISON.json。\n')


# Reuse the unchanged numeric renderer. The sole literal replacement corrects
# its reporting metadata; actual weights already come from the6500 protocol.
original_render_source=inspect.getsource(base.render)
needle='step=6000,seed=p[\'seed\']'
if original_render_source.count(needle)!=1:raise ValueError('Expected exactly one metadata-only step field')
adapted_render_source=original_render_source.replace(needle,'step=6500,seed=p[\'seed\']')
exec(compile(adapted_render_source,str(SELF)+':inherited_render','exec'),base.__dict__)
base.install=install;base.prepare=prepare;base.final=final
original_queue_source=inspect.getsource(base.queue)
adapted_queue_source=original_queue_source.replace('waiting_upstream6309_complete','waiting_step6000_complete').replace('ltx_step=6000','ltx_step=6500')
exec(compile(adapted_queue_source,str(SELF)+':inherited_queue','exec'),base.__dict__)
base.SELF=SELF;base.WORK=WORK;base.UPSTREAM=UPSTREAM;base.TRAIN=TRAIN;base.ARM=ARM


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','queue','worker']);parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage',choices=['cache','render','decode']);parser.add_argument('--rank',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--arms',nargs='+');a=parser.parse_args()
    try:
        if a.action=='prepare':prepare(a.work)
        elif a.action=='queue':base.queue(a.work)
        else:base.worker(a)
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
