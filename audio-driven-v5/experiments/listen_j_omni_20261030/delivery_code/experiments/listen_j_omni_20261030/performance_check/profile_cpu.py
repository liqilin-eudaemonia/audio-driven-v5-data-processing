"""Three existing media samples, CPU-only; preserves production input hashes."""
from pathlib import Path
import sys,time,json,cProfile,pstats,io,hashlib,os
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni import prepare_input
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
import torch
torch.set_num_threads(2)
cfg=read(W/'config.json');checks=read(W/'validation/production_v2_24gpu_audit.json')['checks'];candidates=sorted([read(x['file']) for x in checks if x['node']=='node4'],key=lambda r:r['source']['duration_s']);chosen=[candidates[0],candidates[len(candidates)//2],candidates[-1]]
mc=ModelConfig(model=cfg['model'],trust_remote_code=True,dtype='bfloat16',max_model_len=49152,limit_mm_per_prompt={'audio':1,'video':1})
processor=MULTIMODAL_REGISTRY.create_processor(mc,cache=MultiModalProcessorOnlyCache(mc));reports=[]
for r in chosen:
 origin=read(W/'job_origins'/f"{r['source']['pair']}.json");prof=cProfile.Profile();prof.enable();t=time.monotonic()
 p=prepare_input(dict(sample_id=r['sample_id'],source=r['source']),origin['old_work'],W,cfg['nodes']['node4']['resource_work']);prep=time.monotonic()-t
 assert p['evidence']['video_frames_sha256']==r['input_evidence']['video_frames_sha256'];assert p['evidence']['audio_encoder_sha256']==r['input_evidence']['audio_encoder_sha256']
 times=[];tokens=[]
 for repeat in range(2):
  processor.cache.clear_cache();t=time.monotonic()
  inp=ProcessorInputs(prompt=r['rendered_prompt'],mm_data_items=processor.info.parse_mm_data(dict(video=[p['frames']],audio=[p['audio']])),hf_processor_mm_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=[.25]))
  out=processor.apply(inp,timing_ctx=TimingContext(enabled=False));times.append(time.monotonic()-t);tokens.append(len(out['prompt_token_ids']))
 prof.disable();st=io.StringIO();pstats.Stats(prof,stream=st).sort_stats('cumulative').print_stats(24)
 reports.append(dict(sample_id=r['sample_id'],duration_s=r['source']['duration_s'],input_prepare_s=prep,processor_s=times,prompt_tokens=tokens,existing_generation_seconds=r['seconds'],hashes_identical=True,profile=st.getvalue()))
 print(json.dumps({k:v for k,v in reports[-1].items() if k!='profile'}),flush=True)
write(W/'performance_check/cpu_profile.json',dict(at=time.time(),cpu_threads=2,node=__import__('socket').gethostname(),reports=reports))
