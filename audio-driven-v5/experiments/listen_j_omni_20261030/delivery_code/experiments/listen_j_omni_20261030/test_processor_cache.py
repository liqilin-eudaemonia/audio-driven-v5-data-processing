"""CPU-only regression: repeated silent audio paired with different video frames."""
from pathlib import Path
import json, time
import numpy as np
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
model='/private/ckpts/Qwen3-Omni/Instruct'
cfg=ModelConfig(model=model,trust_remote_code=True,dtype='bfloat16',max_model_len=49152,limit_mm_per_prompt={'audio':1,'video':1})
results=[]
for mode in ("default", "disabled", "reset_between_pairs"):
 enabled=mode!="disabled"
 processor=MULTIMODAL_REGISTRY.create_processor(cfg,cache=MultiModalProcessorOnlyCache(cfg) if enabled else None)
 hf=processor.info.get_hf_processor()
 prompt=hf.apply_chat_template([dict(role='user',content=[dict(type='text',text='Describe the listener.'),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 cases=[]
 for n in range(3):
  if mode=="reset_between_pairs":processor.cache.clear_cache()
  media={'video':[np.full((24,112,112,3),n*40,dtype=np.uint8)],'audio':[np.zeros(48000,dtype=np.float32)]}
  inp=ProcessorInputs(prompt=prompt,mm_data_items=processor.info.parse_mm_data(media),hf_processor_mm_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=[.25]))
  try:
   out=processor.apply(inp,timing_ctx=TimingContext(enabled=False))
   cases.append(dict(case=n,success=True,tokens=len(out['prompt_token_ids'])))
  except Exception as e:
   causes=[];x=e
   while x is not None:causes.append(dict(type=type(x).__name__,message=str(x)[-500:]));x=x.__cause__
   cases.append(dict(case=n,success=False,causes=causes))
 results.append(dict(mode=mode,cache_enabled=enabled,cases=cases))
report=dict(at=time.time(),results=results,passed=all(x['success'] for x in results[2]['cases']) and any(not x['success'] for x in results[0]['cases']))
Path(__file__).with_name('validation').joinpath('processor_cache_regression.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
assert report['passed'], report
