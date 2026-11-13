"""CPU-only: paired request UUIDs must equal reset-cache processor inputs."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
W=Path(__file__).resolve().parents[1]
mc=ModelConfig(model='/private/ckpts/Qwen3-Omni/Instruct',trust_remote_code=True,dtype='bfloat16',max_model_len=49152,limit_mm_per_prompt={'audio':1,'video':1})
def canonical(x):
 if isinstance(x,torch.Tensor):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
 if isinstance(x,np.ndarray):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.tobytes()).hexdigest())
 if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
 if hasattr(x,'data') and not isinstance(x,(str,bytes)):return canonical(x.data)
 if x is None or isinstance(x,(str,bool,int,float)):return x
 raise TypeError(type(x))
results={}
for mode in ['reset','paired_uuid']:
 proc=MULTIMODAL_REGISTRY.create_processor(mc,cache=MultiModalProcessorOnlyCache(mc));hf=proc.info.get_hf_processor()
 prompt=hf.apply_chat_template([dict(role='user',content=[dict(type='text',text='Describe the listener.'),dict(type='video',video='listener://decoded')])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 out=[]
 for n in range(5):
  if mode=='reset':proc.cache.clear_cache()
  mm=dict(video=[np.full((24,112,112,3),(n%3)*40,dtype=np.uint8)],audio=[np.zeros(48000,dtype=np.float32)])
  inp=ProcessorInputs(prompt=prompt,mm_data_items=proc.info.parse_mm_data(mm),mm_uuid_items={'audio':[f'paired-{n}'],'video':[f'paired-{n}']} if mode=='paired_uuid' else None,hf_processor_mm_kwargs=dict(use_audio_in_video=True,fps=8,do_sample_frames=False,second_per_grid_ts=[.25]))
  r=proc.apply(inp,timing_ctx=TimingContext(enabled=False));data=canonical(r['mm_kwargs']);out.append(dict(tokens=r['prompt_token_ids'],tensor_hashes=data))
 results[mode]=out
assert results['reset']==results['paired_uuid']
report=dict(passed=True,at=time.time(),samples=5,processor_outputs_identical=True,scope='CPU processor only, GPU engine streaming not validated',fingerprints=[hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest() for x in results['reset']])
(W/'performance_check/paired_uuids.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
