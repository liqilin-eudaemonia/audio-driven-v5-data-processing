"""Single authorized GPU7: Event+Whisper6309, historical fixed dual200 evaluation."""
import argparse,copy,fcntl,json,os,subprocess,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
WORK=ROOT/'experiments/p2_event_whisper6309_dual200_node11_20261107'
PARENT=ROOT/'experiments/p2_combined_whisper_eval_20261106'
TRAIN=ROOT/'experiments/p2_whisper_recompute30_adopted_20261106/event'
ARM='base_event_whisper_6309'


def install():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_recompute_profile_20261106 import install as profile
    profile()


@contextmanager
def runtime():
    install()
    from p2_whisper_ablation.inference import load_model
    from p2_whisper_ablation.checkpoint import checked
    from p2_whisper_ablation.config import source_identity
    expected=source_identity()
    with patch('p2_whisper_av.inference.load_model',load_model),patch('p2_whisper_av.checkpoint.checked',checked),patch('p2_whisper_av.config.source_identity',lambda:dict(expected)):
        yield


def code_identity():
    files=[SELF,ROOT/'src/p2_whisper_recompute_profile_20261106.py',ROOT/'src/p2_whisper_frozen_resume_20261106.py',ROOT/'src/p2_whisper_ablation/inference.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/metrics.py',ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_testsets_eval/shared_items.py']
    return {str(f):sha(f) for f in files}


def prepare(work):
    work.mkdir(exist_ok=True)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        if p['execution_code']!=code_identity():raise ValueError('Evaluation source changed')
        return p
    save(work/'status.json',dict(stage='cpu_preparing',at=time.time()))
    install()
    import torch
    torch.set_num_threads(2)
    from p2_whisper_ablation.checkpoint import checked
    from p2_whisper_ablation.config import source_identity
    from p2_listen.cache import atomic_torch
    from p2_combined_whisper_eval import link
    from p2_testsets_eval.runtime import ready
    original=TRAIN/'step_006309.pt';marker=read(original.with_suffix('.complete.json'))
    target=work/'checkpoints/step_006309.pt';receipt=target.with_suffix('.export.json')
    if receipt.exists():
        audit=read(receipt)
        if audit['original_sha256']!=marker['sha256']:raise ValueError('Original checkpoint changed')
        saved=checked(target)
    else:
        saved=checked(original)
        if saved['identity']!=read(TRAIN/'run.json') or saved['identity']['source']!=source_identity():raise ValueError('Training checkpoint identity differs')
        model_only={k:saved[k] for k in ['schema','step','identity','model']}
        atomic_torch(target,model_only);save(target.with_suffix('.complete.json'),dict(step=6309,sha256=sha(target)))
        exported=checked(target)
        if exported['identity']!=saved['identity'] or set(exported['model'])!=set(saved['model']):raise ValueError('Export metadata differs')
        if any(not torch.equal(t,exported['model'][k]) for k,t in saved['model'].items()):raise ValueError('Export altered weights')
        audit=dict(original_checkpoint=str(original),original_sha256=marker['sha256'],inference_checkpoint=str(target),inference_sha256=read(target.with_suffix('.complete.json'))['sha256'],step=6309,all_model_tensors_bitwise_equal=True,optimizer_omitted=True)
        save(receipt,audit)
    identity=saved['identity'];assert saved['step']==6309 and identity['experiment']['loss']=='av_event'
    assert identity['experiment']['teacher_cache'] is None
    parent=read(PARENT/'protocol.json');items=read(PARENT/'shared_items.json')
    assert sha(PARENT/'shared_items.json')==parent['shared_items_sha256']
    assert len(items)==400 and len({x['key'] for x in items})==400
    for dataset,spec in parent['datasets'].items():
        assert sum(x['dataset']==dataset for x in items)==200
        assert sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
        link(PARENT/'conditions'/dataset,work/'conditions'/dataset)
        link(PARENT/'whisper'/dataset,work/'whisper'/dataset)
    assert read(Path(identity['experiment']['whisper_cache'])/'protocol.json')['encoder']==parent['whisper_encoder']
    for x in items:
        folder=work/'conditions'/x['key'];feature=work/'whisper'/x['key']
        assert all(ready(y) for y in [folder/'qwen',folder/'text',feature])
        plan=read(folder/'qwen/plan.json');meta=read(feature/'metadata.json')
        assert plan['adapter'] is None and plan['target_description_access'] is False
        assert meta['encoder']==parent['whisper_encoder'] and meta['feature_code']==parent['feature_code']
        assert meta['source']['input_audit_sha256']==sha(folder/'input_audit.json')
    for rank in range(8):link(PARENT/'negative/00',work/'negative'/f'{rank:02d}')
    cluster=read(ROOT/'experiments/p2_whisper_continue30_20261106/event/cluster.json')
    node=next(n for n in cluster['nodes'] if n['name']=='node11');node['gpus']=[7]
    cluster['nodes']=[node];cluster['training_script']=str(SELF);save(work/'cluster.json',cluster)
    for rank,shard in enumerate(partition(items,8)):save(work/'shards'/f'{rank:02d}.json',shard)
    save(work/'shared_items.json',items)
    p=copy.deepcopy(parent);p.pop('metrics_manifest_overrides',None)
    p.update(schema='event-whisper6309-single-gpu-dual200-v1',created_at=time.time(),arms=[dict(name=ARM,kind='event_whisper',step=6309,checkpoint=str(target),checkpoint_sha256=audit['inference_sha256'],original_checkpoint=str(original),original_sha256=marker['sha256'],planner=parent['planner'])],checkpoint_identity=identity,
             workers=8,total_outputs=400,total_samples=400,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(SELF),
             shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
             authorized_scope='node11 172.29.189.119 physical GPU7 only; user permits overlay; no other tasks interrupted',inference_nodes=['node11'],
             metrics_workers_per_gpu=4,metrics_use_released_inference_nodes=False,
             conditioning='historical base Qwen H/Gemma plus exact speaker-only Whisper; AV+Event trained model, no Feature',
             comparison='6309 updates: first4000 batch31 then2309 batch30 =193270 sample exposures; historical32x6000=192000 (+0.66%)',
             execution_order='reuse400 source-only conditions and Whisper -> resident single GPU render -> resident VAE decode -> native four-way metrics -> full200 feature merge',execution_code=code_identity())
    # Scope GT reuse to this exact historical dual200 cache. No broad corpus scan.
    p['metrics_gt_cache']=str(PARENT/'gt_cache')
    save(work/'protocol.json',p)
    save(work/'CPU_PRECHECK.json',dict(success=True,at=time.time(),datasets=parent['datasets'],outputs=400,reused_conditions=400,source_only=True,base_qwen=True,physical_gpu=7,host='172.29.189.119',generation=p['generation'],seed=p['seed'],export=audit,reference_model=parent['arms'][0]['name'],smoke='pending'))
    return p


def worker(work,stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise ValueError('Expected physical GPU7 only')
    with runtime():
        import torch
        torch.set_num_threads(4)
        if torch.cuda.device_count()!=1:raise ValueError('Expected one visible device')
        save(work/'device.json',dict(host=os.uname().nodename,physical_gpu=7,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),device=torch.cuda.get_device_name(0),stage=stage,at=time.time()))
        from p2_testsets_eval.whisper_runtime import worker as native
        native(work,stage,0)


def smoke_gate(work,p):
    from p2_testsets_eval.runtime import ready,probe
    from p2_testsets_eval.metrics import prepare_smoke
    for dataset in p['datasets']:
        chosen=[x for x in read(work/'shards/00.json')[:2]+read(work/'shards/01.json')[:2] if x['dataset']==dataset][:2]
        assert len(chosen)==2
        for x in chosen:
            out=work/'generated'/ARM/x['key'];assert ready(out)
            m=read(out/'metadata.json');g=x['geometry'];info=probe(out/'generated.mp4')
            assert m['step']==6309 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
            assert not m['target_description_access'] and not m['listener_future_input']
            assert info['frames']==g['frames'] and info['width']==info['height']==512
    save(work/'GPU_SMOKE_COMPLETE.json',dict(success=True,step=6309,outputs_checked=4,datasets=list(p['datasets']),source_only=True,at=time.time()))
    save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=4,at=time.time()))
    prepare_smoke(work)


def final_audit(work,p):
    from p2_testsets_eval.runtime import ready
    from p2_testsets_eval.protocol import rows
    results=read(work/'RESULTS.json');summaries=[]
    for dataset,spec in p['datasets'].items():
        items=[x for x in read(work/'shared_items.json') if x['dataset']==dataset]
        for x in items:
            out=work/'generated'/ARM/x['key'];assert ready(out)
            m=read(out/'metadata.json');assert m['step']==6309 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
            assert not m['target_description_access'] and not m['listener_future_input']
        s=results[ARM+'/'+dataset];assert s['requested_pairs']==200
        assert s['audio']['requested_pairs']==200 and s['audio']['pairs_with_errors']==0
        assert all(s['coverage'][k]['pairs']==200 for k in ['psnr','ssim','fid','fvd','lpips'])
        summaries.append(dict(dataset=dataset,outputs=len(items),metrics=s['metrics'],coverage=s['coverage'],audio=s['audio']))
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=400,groups=summaries,physical_gpu=7))
    historical=read(PARENT/'RESULTS.json');save(work/'COMPARISON.json',dict(current=results,historical=historical,notes=[p['comparison'],'Same fixed200 per dataset, seed137, Euler30, CFG3/7; full200 FID/FVD, coverage preserved']))
    lines=['# Event＋Whisper 6309步：固定双200条','','仅node11 GPU7；固定ResponseNet分层随机200＋Seamless固定200，seed137／Euler30／CFG3、7。',p['comparison'],'','| 数据集 | PSNR↑ | SSIM↑ | FID↓ | FVD↓ | LPIPS↓ | Emo-Acc↑ | Emo-Div | ΔResp↓ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in p['datasets']:
        m=results[ARM+'/'+d]['metrics'];values=[m.get(k) for k in ['psnr','ssim','fid','fvd','lpips','emo_acc','emo_div_generated','delta_resp']]
        lines.append('| '+d+' | '+' | '.join('N/A' if v is None else f'{v:.6g}' for v in values)+' |')
    lines+=['','完整覆盖、音频活动、UTMOS有效性和逐条结果见RESULTS.json与metrics/；不把静音／短回应的N/A填零。']
    (work/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def run(work):
    work.mkdir(exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        save(work/'CONTROLLER.json',process_record());p=prepare(work)
        if (work/'FINAL_AUDIT.json').exists():return
        for stage in ['render','decode']:
            if (work/'STOP').exists():save(work/'status.json',dict(stage='stopped',at=time.time()));return
            marker=work/'stages'/(stage+'.json')
            if marker.exists():continue
            save(work/'status.json',dict(stage=stage,at=time.time(),total_outputs=400))
            for attempt in range(3):
                with (work/(stage+'.log')).open('ab',buffering=0) as log:
                    code=subprocess.call([str(PYTHON),'-u',str(SELF),'worker','--work',str(work),'--stage',stage],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if code==0:break
                if attempt==2:raise RuntimeError(f'{stage} failed; see {stage}.log')
                time.sleep(5)
            if (work/'STOP').exists():return
            save(marker,dict(complete=True,at=time.time()))
        from p2_testsets_eval.runtime import ready
        assert all(ready(work/'generated'/ARM/x['key']) for x in read(work/'shared_items.json'))
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time()))
        from p2_testsets_eval.cluster import bind
        bind(work);smoke_gate(work,p)
        save(work/'status.json',dict(stage='metrics',at=time.time()))
        from p2_testsets_eval.metrics import queue
        queue(work)
        if (work/'STOP').exists():return
        final_audit(work,p);save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,at=time.time(),training_resumed=False))
        save(work/'status.json',dict(stage='complete',at=time.time(),outputs=400))


def launch(work):
    # Must execute on the authorized machine. Only its selected GPU is checked.
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:raise ValueError('Launch on node11 only')
    used,util=[float(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    if used>14000:raise RuntimeError('GPU7 lacks memory headroom for authorized overlay; no eviction')
    work.mkdir(exist_ok=True)
    if (work/'CONTROLLER.json').exists():
        record=read(work/'CONTROLLER.json')
        try:
            s=Path('/proc',str(record['pid']),'stat').read_text().split()
            if s[21]==record['start_ticks'] and s[2]!='Z':raise RuntimeError('Existing evaluation controller still live')
        except FileNotFoundError:pass
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    for k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','P2_TRAIN_MEMORY_GIB']:env.pop(k,None)
    with (work/'controller.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([str(PYTHON),'-u',str(SELF),'run','--work',str(work)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(work/'LAUNCH.json',dict(pid=proc.pid,host=os.uname().nodename,physical_gpu=7,at=time.time()));print('launched',proc.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','run','worker']);parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage',choices=['render','decode']);a=parser.parse_args()
    try:
        if a.action=='prepare':prepare(a.work)
        elif a.action=='launch':launch(a.work)
        elif a.action=='worker':worker(a.work,a.stage)
        else:run(a.work)
    except Exception as exc:
        a.work.mkdir(exist_ok=True);save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
