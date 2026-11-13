"""Checkpoint-native rendering with an explicitly autonomous condition bundle."""
from contextlib import contextmanager
from pathlib import Path
import json,subprocess,sys,uuid
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json,observation
from .protocol import lock,open_protocol
from .runtime import Reader

@contextmanager
def condition_loaders(native,c,bundle):
    from p2_listen.cache import tree_to
    old_input,old_text=native.load_input,native.load_text
    def load_input(config,obs,device='cpu'):
        if config!=c or obs['sample_id']!=bundle['plan']['sample_id']:raise ValueError('Unexpected native input request')
        return bundle['condition'].to(device),tree_to(bundle['layout'],device)
    def load_text(config,device='cpu'):
        if config!=c:raise ValueError('Unexpected native text request')
        return tree_to(bundle['text'],device)
    native.load_input,native.load_text=load_input,load_text
    try:yield
    finally:native.load_input,native.load_text=old_input,old_text


def legacy(job,reader,obs,out,device):
    from p2_guided_prompt_en.runtime import historical_sources
    trained=read(Path(job['run'])/'run.json');c=trained['config']
    if Path(job['checkpoint']).resolve().parent!=Path(job['run']).resolve():raise ValueError('Checkpoint/run mismatch')
    with historical_sources(job['run'],job['restoration']) as restored:
        from p2_listen import generation as native
        bundle=reader.conditions(job['sample_id'],c,device)
        release=dict(config_sha256=digest(c),checkpoint_sha256=[sha(job['checkpoint'])],sample_ids=[job['sample_id']],
                     seeds=[job['seed']],evaluation_role='autonomous_two_pass',target_annotation_access=False)
        path=reader.work/'releases'/(digest([job,release])+'.json');atomic_json(path,release,immutable=True)
        with condition_loaders(native,c,bundle):
            metadata=native.generate(c,obs,out,checkpoint=job['checkpoint'],seed=job['seed'],device=device,release=path)
        if metadata['step']!=c['training']['steps']:raise ValueError('Final legacy checkpoint required')
        return dict(metadata,historical_source_restoration=restored),bundle['plan']


def matrix(job,reader,obs,out,device):
    import torch
    sys.path.append(str(ROOT/'scripts'))
    from p2_duration_execution import install_backend
    from p2_matrix16.checkpoint import checked
    from p2_matrix16.model import MatrixModel
    from p2_listen.config import extractor_config
    from p2_listen.features import weights_identity,make_observation,unload
    from p2_listen.cache import atomic_torch,cpu
    from p2_listen.layout import geometry
    from p2_listen.runtime import guided_sample,append_negative
    backend=install_backend();saved=checked(job['checkpoint']);identity=saved['identity'];c=identity['experiment']
    if c['stage']!=1 or c['text_condition']!='guided':raise ValueError('Only frozen-Qwen guided Stage1 supported; Stage2 requires live LoRA hidden')
    if saved['step']!=c['training']['steps']:raise ValueError('Final checkpoint required')
    if identity['source']!=backend.source_identity():raise ValueError('Checkpoint source changed')
    d=backend.data_config(c);cfg=extractor_config(d)
    if identity['assets']!={k:weights_identity(d,k) for k in ('omni','ltx','gemma')}:raise ValueError('Native assets changed')
    bundle=reader.conditions(job['sample_id'],d,device);plan=bundle['plan']
    from omni_av_p2.ltx import load_renderer,append_conditions,unpack_latents
    from omni_av_p2.inference import reference_latents
    from omni_av.ltx_runtime import with_text
    from omni_av_p3_staged.media import decode_and_save
    out.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        g=geometry(obs);av=reference_latents(cfg,make_observation(obs),g,device)
        renderer,strategy,sampler=load_renderer(cfg,device);model=MatrixModel(c,cfg,renderer).to(device)
        if model.text_targets!=identity['text_targets'] or model.qwen_targets!=identity['qwen_targets']:raise ValueError('Adapter targets changed')
        model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False);renderer.eval().requires_grad_(False)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            u=model.connector(bundle['condition']);logits=model.event_heads(u,bundle['layout'],'probe_only')
        atomic_torch(out/'probe.pt',cpu(dict(logits=logits,layout=bundle['layout'])))
        text=bundle['text'];del model,saved,bundle,logits;unload()
        torch.manual_seed(job['seed']);prepared=strategy.prepare_training_inputs(with_text(av,text['positive']),sampler)
        prepared=append_conditions(prepared,u);negative=append_negative(text['negative'],u)
        torch.manual_seed(job['seed']);params=d['generation']
        sampled=guided_sample(renderer,prepared,negative,params['steps'],params['video_cfg'],params['audio_cfg'])
        video,audio=unpack_latents(sampled,av)
        del renderer,prepared,negative,sampled,av,u,text;unload()
        decoded=decode_and_save(cfg,video,audio,g['fps'],out/'padded.mp4',device,raw_audio_path=out/'raw_audio.pt')
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'padded.mp4'),'-t',str(obs['requested_duration_s']),
                    '-c:v','libx264','-crf','18','-c:a','aac',str(out/'generated.mp4')],check=True)
    import soundfile as sf
    raw=torch.load(out/'raw_audio.pt',map_location='cpu',weights_only=True);wave=raw['wave'];count=round(obs['requested_duration_s']*raw['rate'])
    wave=torch.nn.functional.pad(wave,(0,max(0,count-wave.shape[-1])))[...,:count]
    sf.write(out/'listener.wav',wave.T.numpy(),raw['rate'],subtype='FLOAT')
    return dict(status='success',sample_id=job['sample_id'],sample_seed=job['seed'],step=c['training']['steps'],
                geometry=g,decoded=decoded,generation=d['generation']),plan


def generate(job,device='cuda:0'):
    import torch
    from p2_glare_eval.common import artifact_valid,complete
    torch.set_num_threads(2)
    reader=Reader(job['work']);p=reader.protocol
    if job not in p['jobs']:raise ValueError('Generation job outside frozen protocol')
    obs=observation(next(o for o in p['observations'] if o['sample_id']==job['sample_id']))
    checkpoint=Path(job['checkpoint']);h=sha(checkpoint)
    if read(checkpoint.with_suffix('.complete.json'))['sha256']!=h:raise ValueError('Checkpoint differs from completion marker')
    plan=reader.plan(job['sample_id'])
    ident=dict(protocol_sha256=digest(p),job=job,checkpoint_sha256=h,plan_sha256=digest(plan),
               evaluation_role='autonomous_two_pass',target_annotation_access=False,gt_media_access=False)
    out=Path(job['output'])
    with lock(out.with_suffix('.lock')):
        if artifact_valid(out,ident):return dict(status='reused',output=str(out))
        if out.exists():raise ValueError('Output exists with no matching completion marker; inspect before retry')
        attempt=reader.work/'generation_attempts'/job['arm']/(job['sample_id']+'__'+uuid.uuid4().hex)
        try:
            metadata,used=(legacy if job['family']=='legacy' else matrix)(job,reader,obs,attempt,device)
            if used!=plan:raise ValueError('Generation plan changed')
            streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(attempt/'generated.mp4')],text=True))['streams']
            if {s['codec_type'] for s in streams}!={'video','audio'}:raise ValueError('Missing AV stream')
            metadata.update(plan_source='qwen_autonomous',evaluation_role='autonomous_two_pass',gt_access=False,
                target_annotation_access=False,gt_media_access=False,qwen_frozen=True,qwen_lora=False,
                qwen_hidden_scope='all_input_and_plan_tokens',prompt_sha256=plan['prompt_sha256'],
                plan_sha256=digest(plan),protocol_sha256=digest(p),checkpoint_sha256=h,
                conditioning='same autonomous plan in Qwen hidden and Gemma',sample_seed=job['seed'])
            atomic_json(attempt/'completed.json',metadata);atomic_json(attempt/'plan.json',plan)
            (attempt/'prompt.txt').write_text(plan['prompt'])
            complete(attempt,ident,[attempt/n for n in ('generated.mp4','listener.wav','completed.json','prompt.txt','plan.json')],
                     target_annotation_access=False,gt_media_access=False)
            out.parent.mkdir(parents=True,exist_ok=True);attempt.rename(out)
            return dict(status='success',output=str(out))
        except Exception as e:
            atomic_json(attempt/'failed.json',dict(error_type=type(e).__name__,error=str(e),job=job));raise
