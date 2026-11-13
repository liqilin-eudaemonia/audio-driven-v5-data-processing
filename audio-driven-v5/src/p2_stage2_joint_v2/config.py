"""Stage2 metadata is independent of the checkpointed Stage1 training config."""
from pathlib import Path
import copy,math
from p2_listen.contracts import ROOT,read,sha
from p2_current_component_ablation.config import native_whisper_view
from . import VERSION


def flags(c):
    return c['components']


def source_identity():
    from p2_stage2_joint.config import source_identity as previous
    result=previous()
    from p2_ltx_all_lora_components import lora
    result[str(Path(lora.__file__))]=sha(lora.__file__)
    for p in Path(__file__).parent.glob('*.py'):result[str(p)]=sha(p)
    for name in ('p2_planning_sft/model.py','p2_planning_sft/data.py','p2_plan_cache/encoding.py',
                 'p2_testsets_eval/runtime.py','p2_whisper_av/features.py'):
        p=ROOT/'src'/name;result[str(p)]=sha(p)
    cli=ROOT/'scripts/p2_stage2_joint_v2_train.py'
    if cli.exists():result[str(cli)]=sha(cli)
    worker=ROOT/'scripts/p2_stage2_v2_eval_worker.py'
    if worker.exists():result[str(worker)]=sha(worker)
    for folder in (ROOT/'src/p2_event_presence',ROOT.parent/'OmniResponse/evaluation/listener_metrics',ROOT.parent/'OmniResponse/evaluation/video/paired_metrics'):
        for p in folder.glob('*.py'):result[str(p)]=sha(p)
    for p in (ROOT/'scripts/j_upright_response_sft_data.py',ROOT/'scripts/p2_event_presence.py'):
        result[str(p)]=sha(p)
    return result


def validate(c,verify_payloads=False):
    if c.get('schema')!=VERSION:raise ValueError('Unknown joint alignment configuration')
    if c['steps']!=300 or c['milestones']!=[0,100,300]or c['global_batch']!=31 or c['warmup_steps']!=20:
        raise ValueError('Require agreed300-step trial and fixed0/100/300 checkpoints')
    if c['qwen_lr']<=0 or not 0<c['generation_lr_scale']<=1 or c['planning_weight']<=0:
        raise ValueError('Invalid alignment learning rates/loss')
    if c['validation_size']!=64:raise ValueError('Require64 independent validation samples')
    if any(type(c[k])not in (int,float)or not math.isfinite(c[k])for k in ('qwen_lr','generation_lr_scale','planning_weight')):
        raise ValueError('Finite numeric alignment coefficients required')
    if sha(c['parent_config'])!=c['parent_config_sha256']:raise ValueError('Parent configuration changed')
    parent=read(c['parent_config']);g=c['generator']
    if g.get('ltx_lora_scope')!=parent.get('ltx_lora_scope'):raise ValueError('Selected Stage1 LoRA scope changed')
    spec=c.get('calibration',{})
    if (spec.get('samples') not in range(8,17) or type(spec.get('samples'))is not int or spec.get('target_ratio')!=.2 or spec.get('reference_band')!=[.1,.3] or spec.get('max_global_ratio')!=1. or spec.get('chunk_elements')!=2097152 or type(spec.get('seed'))is not int):raise ValueError('Invalid frozen gradient calibration policy')
    expected=dict(whisper=bool(parent.get('whisper')),feature='feature'in parent['loss'],event='event'in parent['loss'])
    if g['components']!=expected or g['loss_weights']!=parent['loss_weights']:raise ValueError('Selected Stage1 components/objective changed')
    if Path(c['output']).resolve()==Path(parent['output']).resolve():raise ValueError('Separate Stage2 output required')
    for key in ('connector','text_lora'):
        if g[key]!=parent[key]:raise ValueError('Selected Stage1 architecture changed: '+key)
    if expected['whisper']and g['whisper']!=parent['whisper']:raise ValueError('Selected Whisper branch changed')
    baseline=read(ROOT/'experiments/p2_current500h_av_whisper31_20261107/config.json')
    for key in ('j_cache','current_cache_index','sample_list','duration_sampling','execution','event_supervision'):
        if g[key]!=baseline[key]:raise ValueError('Final500h data/schedule policy changed: '+key)
    if 'calibration_manifest' in c:
        m=read(c['calibration_manifest'])
        if sha(c['calibration_manifest'])!=c['calibration_manifest_sha256']or len(m['samples'])!=c['calibration']['samples']or not m['training_only']or m['test_used']:raise ValueError('Calibration membership changed')
    if 'policy' in c and sha(c['policy'])!=c['policy_sha256']:raise ValueError('Predeclared trial policy changed')
    if 'validation_receipt' in c:
        if sha(c['validation_receipt'])!=c['validation_receipt_sha256']:raise ValueError('Validation receipt changed')
        receipt=read(c['validation_receipt'])
        for name,receipt_key in [('validation_manifest','validation_sha256'),('planning_training_ids','planning_ids_sha256'),('planning_caption_hashes','planning_caption_hashes_sha256')]:
            if c[name+'_sha256']!=receipt[receipt_key]or sha(c[name])!=c[name+'_sha256']:raise ValueError('Validation/eligibility binding changed')
    f=c['generator']['components']
    if set(f)!={'whisper','feature','event'}or any(type(v)is not bool for v in f.values()):raise ValueError('Invalid component flags')
    if c['generator']['stage']!=1 or c['generator']['qwen_lora']['enabled']:raise ValueError('Generator must not build a second Qwen adapter')
    if c['generator']['output']!=c['output']:raise ValueError('Generator/output mismatch')
    if f['event']and c['generator']['hidden_event_supervision']!=dict(readout='last_block_before_output',clip_weight=1.,temporal_weight=0.):
        raise ValueError('Only current LTX whole-clip Event is supported')
    for key in ('stage1_checkpoint','adapter_config','adapter_weights','planning_complete'):
        asset=c['assets'][key];path=Path(asset['path']);stat=path.stat()
        if stat.st_size!=asset['bytes']or stat.st_mtime_ns!=asset['mtime_ns']:
            raise ValueError('Frozen asset metadata changed: '+key)
        if verify_payloads and sha(path)!=asset['sha256']:raise ValueError('Frozen asset changed: '+key)
    complete=read(c['assets']['planning_complete']['path'])
    if not complete['success']or complete['global_step']!=2752:raise ValueError('Require completedSFT2752')
    evaluation=c.get('evaluation')
    if evaluation is not None:
        if evaluation['seed']!=137 or evaluation['sampling']!=dict(steps=30,video_cfg=3.,audio_cfg=7.)or evaluation['max_new_tokens']!=512 or evaluation['planner_max_pixels']!=50176:
            raise ValueError('Frozen trial generation settings changed')
    return c


def build_config(parent_config,checkpoint,adapter,output,planning_weight=0.0001):
    """Explicit parent selection is required; no default checkpoint is silently chosen."""
    parent=read(parent_config);adapter=Path(adapter).resolve();checkpoint=Path(checkpoint).resolve()
    full=read(ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106/config.json')
    base=read(ROOT/'experiments/p2_current500h_av_whisper31_20261107/config.json')
    if parent.get('current_cache_index')!=base['current_cache_index']:raise ValueError('Parent must use final500h')
    f=dict(whisper=bool(parent.get('whisper')),feature='feature' in parent['loss'],event='event' in parent['loss'])
    if f['event']and parent.get('hidden_event_supervision')!=full['hidden_event_supervision']:
        raise ValueError('Historical Connector Event parents are not compatible')
    g=copy.deepcopy(base)
    if parent.get('ltx_lora_scope') is not None:g['ltx_lora_scope']=copy.deepcopy(parent['ltx_lora_scope'])
    loss='av_event_feature'if f['event']and f['feature']else'av_event'if f['event']else'av_feature'if f['feature']else'av'
    g.update(components=f,loss=loss,id='guided_'+loss+'_s1',event_head_training=f['event'],output=str(Path(output).resolve()),
        teacher_cache=parent.get('teacher_cache')if f['feature']else None,
        hidden_event_supervision=dict(readout='last_block_before_output',clip_weight=1. if f['event']else 0.,temporal_weight=0.),loss_weights=copy.deepcopy(parent['loss_weights']))
    if not f['whisper']:
        for key in ('whisper','whisper_cache'):g.pop(key,None)
    for key in ('clip_labels','clip_numeric_labels','current_source'):g[key]=full[key]if f['event']else None
    if g['connector']!=parent['connector']:raise ValueError('Parent Connector architecture differs')
    marker=read(checkpoint.with_suffix('.complete.json'))
    assets={key:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)for key,path in {
        'stage1_checkpoint':checkpoint,'adapter_config':adapter/'adapter_config.json',
        'adapter_weights':adapter/'adapter_model.safetensors','planning_complete':adapter.parent/'COMPLETE.json'}.items()}
    if marker['sha256']!=assets['stage1_checkpoint']['sha256']:raise ValueError('Incomplete or corrupt Stage1 checkpoint')
    evaluation=dict(seed=137,sampling=dict(steps=30,video_cfg=3.,audio_cfg=7.),max_new_tokens=512,planner_max_pixels=50176,
        metric_python='/private/software/conda/envs/AnyTalker/bin/python',
        metric_assets=str(ROOT.parent/'OmniResponse/evaluation/video/paired_metrics/assets.local.json'),
        event_config=str(ROOT/'configs/p2_event_presence_v1.json'))
    return validate(dict(schema=VERSION,generator=g,output=g['output'],steps=300,milestones=[0,100,300],global_batch=31,evaluation=evaluation,
        warmup_steps=20,qwen_lr=1e-6,generation_lr_scale=1/3,planning_weight=planning_weight,validation_size=64,calibration=dict(samples=16,seed=20261112,target_ratio=.2,reference_band=[.1,.3],max_global_ratio=1.,chunk_elements=2097152),
        parent_config=str(Path(parent_config).resolve()),parent_config_sha256=sha(parent_config),assets=assets))
