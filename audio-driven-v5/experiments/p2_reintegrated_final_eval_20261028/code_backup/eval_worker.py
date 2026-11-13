"""Isolated inference adapter; preserves native training/checkpoint identities."""
import sys, json, os, subprocess, fcntl
from contextlib import contextmanager
from pathlib import Path
from eval_common import HERE,ROOT,read,write,sha,english_extension
PLANS=Path(os.environ.get('P2_EVAL_PLAN_WORK',str(HERE/'plans')))

@contextmanager
def blocking_cache_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield

def memory_limit():
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(64*1024**3/torch.cuda.get_device_properties(0).total_memory)

def bootstrap(run=None):
    if run:
        import runpy
        runpy.run_path(str(ROOT/'scripts/p2_eval_generate.py'))['bootstrap'](['--run',run])
    sys.path.insert(0,str(ROOT/'src'));english_extension()

def legacy(job):
    bootstrap(job['run']);memory_limit()
    from p2_listen.config import activate
    activate(read(Path(job['run'])/'run.json')['config'])
    import p2_glare_eval.generation as g
    # The outer controller verifies headroom and yields before training handoff;
    # retain a concrete per-process cap instead of requiring a vacant device.
    g.check_idle_device=lambda device: memory_limit()
    from p2_guided_prompt_en.runtime import generate
    generate(HERE/'legacy_prompts',job['run'],job['checkpoint'],job['sample_id'],job['output'],
        seed=job['seed'],restoration=job['restoration'])

def matrix(job):
    bootstrap();memory_limit()
    import torch
    import p2_guided_chain.backend
    sys.path.append(str(ROOT/'scripts'))
    from p2_duration_execution import install_backend
    backend=install_backend()
    from p2_matrix16.checkpoint import checked
    from p2_matrix16.model import MatrixModel
    from p2_listen.config import extractor_config
    from p2_listen.features import weights_identity,make_observation,unload
    from p2_listen.contracts import observation,digest,atomic_json
    from p2_listen.cache import atomic_torch,cpu
    from p2_plan_cache.reader import PlanCacheReader
    from p2_listen.layout import geometry
    from p2_listen.runtime import guided_sample,append_negative
    from p2_glare_eval.common import complete
    saved=checked(job['checkpoint']);identity=saved['identity'];c=identity['experiment']
    if c['stage']!=1 or c['text_condition']!='guided':raise ValueError('Expected frozen-Qwen guided Stage1')
    if saved['step']!=c['training']['steps']:raise ValueError('This evaluation requires final checkpoint; intermediate step200 is forbidden')
    if identity['source']!=backend.source_identity():raise ValueError('Checkpoint source changed')
    d=backend.data_config(c);cfg=extractor_config(d)
    if identity['assets']!={k:weights_identity(d,k) for k in ('omni','ltx','gemma')}:raise ValueError('Native assets changed')
    plan_work=PLANS
    warm=HERE/'cluster24/train_warm'
    if PLANS==HERE/'plans' and job['split']=='train24' and (warm/'protocol.json').exists():
        # Exact same native cache identities; allow train diagnostics to start
        # while independent test annotations are still being consolidated.
        plan_work=warm
    reader=PlanCacheReader(plan_work);item=read(plan_work/'samples'/(job['sample_id']+'.json'))
    obs=observation(item['observation']);bundle=reader.conditions(job['sample_id'],'A2','cuda:0')
    if bundle['text']['positive']['prompt']!=item['guidance']['prompt']:raise ValueError('Plan/text mismatch')
    from omni_av_p2.ltx import load_renderer,append_conditions,unpack_latents
    from omni_av_p2.inference import reference_latents
    from omni_av.ltx_runtime import with_text
    from omni_av_p3_staged.media import decode_and_save
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    metadata=dict(sample_id=job['sample_id'],sample_seed=job['seed'],step=saved['step'],checkpoint_sha256=sha(job['checkpoint']),
        observation_sha256=digest(obs),plan_sha256=item['guidance']['prompt_sha256'],plan_source='annotation_oracle',
        guidance_protocol_sha256=digest(reader.protocol),original_annotation_sha256=digest(item['original_record']),
        qwen_hidden_scope='all_input_and_plan_tokens',
        target_annotation_access=True,gt_media_access=False,qwen_frozen=True,conditioning='same plan in Qwen hidden and Gemma',
        evaluation_role='mid_training_diagnostic' if saved['step']==200 else 'final_annotation_oracle',generation=d['generation'])
    try:
        with torch.inference_mode():
            g=geometry(obs);av=reference_latents(cfg,make_observation(obs),g,'cuda:0')
            renderer,strategy,sampler=load_renderer(cfg,'cuda:0')
            model=MatrixModel(c,cfg,renderer).to('cuda:0')
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
            decoded=decode_and_save(cfg,video,audio,g['fps'],out/'padded.mp4','cuda:0',raw_audio_path=out/'raw_audio.pt')
        subprocess.run(['ffmpeg','-v','error','-i',str(out/'padded.mp4'),'-t',str(obs['requested_duration_s']),'-c:v','libx264','-crf','18','-c:a','aac',str(out/'generated.mp4')],check=True)
        import soundfile as sf
        raw=torch.load(out/'raw_audio.pt',map_location='cpu',weights_only=True);wave=raw['wave'];count=round(obs['requested_duration_s']*raw['rate'])
        wave=torch.nn.functional.pad(wave,(0,max(0,count-wave.shape[-1])))[...,:count]
        sf.write(out/'listener.wav',wave.T.numpy(),raw['rate'],subtype='FLOAT')
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/'generated.mp4')],text=True))['streams']
        assert any(s['codec_type']=='audio' for s in streams),'Missing audio track'
        metadata.update(status='success',geometry=g,decoded=decoded,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        atomic_json(out/'completed.json',metadata);(out/'prompt.txt').write_text(item['guidance']['prompt'])
        complete(out,job,[out/n for n in ['generated.mp4','listener.wav','completed.json','prompt.txt']],gt_media_access=False,target_annotation_access=True)
    except Exception as exc:
        write(out/'failed.json',dict(error=repr(exc)));raise

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='cache':
        bootstrap();memory_limit()
        import p2_plan_cache.runtime as runtime
        runtime.lock=blocking_cache_lock
        value=runtime.cache(PLANS,sys.argv[2],sample_ids=read(sys.argv[3]));assert value['success'],value
    elif cmd=='legacy-cache':
        run=sys.argv[2];bootstrap(run);memory_limit()
        from p2_listen.config import activate
        activate(read(Path(run)/'run.json')['config'])
        from p2_guided_prompt_en.runtime import cache_text
        result=cache_text(HERE/'legacy_prompts',run,read(sys.argv[3]));assert result['success'],result
    elif cmd=='generate':
        job=read(sys.argv[2]);legacy(job) if job['family']=='legacy' else matrix(job)
    else:raise ValueError(cmd)
