"""Native cache implementation with explicit recording of length-limited model outputs.
Only difference: after the unchanged 512/1024-token native generation and retry,
keep the actual 1024-token output if no EOS was generated; never synthesize EOS,
rewrite repetition, use GT text, or drop a difficult sample. Native extraction,
Gemma, prompts, dtypes, audio/frame preparation and all scoring stay unchanged.
Imported inside the worker after inference config/progress bindings are installed.
"""
from p2_testsets_eval.runtime import *

def cache(work,items,rank,p):
 import torch
 from p2_listen.cache import atomic_torch,cpu
 from p2_listen.features import unload
 from p2_planning_sft.data import PlanningCollator,SYSTEM,USER
 from p2_j_cache.data import caption_prompt
 from p2_plan_cache.encoding import extract_planned_input,encode_text
 from p3_listen.prompts import FixedPrompt
 d,cfg=data_config();device='cuda:0'
 qtodo=[x for x in items if not ready(work/'conditions'/x['key']/'qwen')]
 if qtodo:
  from omni_av_p2.data import load_omni
  adapter=planner_identity(p)
  if adapter:
   from p2_j_sft_hidden import Reader,load_session
   reader=Reader(p['sft_hidden_cache'])
   if reader.protocol['adapter']!=adapter:raise ValueError('Inference LoRA differs from cached training H')
   runtime,processor,peft=load_session(reader)
  else:runtime,processor=load_omni(cfg,device)
  collator=PlanningCollator(cfg.omni_path);collator.processor=processor
  processor.tokenizer.padding_side='left';eos=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
  for start in range(0,len(qtodo),p['planner_batch']):
   batch=qtodo[start:start+p['planner_batch']]
   with ThreadPoolExecutor(max_workers=4) as pool:
    raw=list(pool.map(lambda x:conditioning(x,work/'conditions'/x['key']),batch))
   inputs=collator.source_inputs(raw).to(device,dtype=torch.bfloat16)
   runtime.thinker.rope_deltas=None
   with torch.inference_mode():
    generated=runtime.thinker.generate(**inputs,max_new_tokens=p['planner_max_new_tokens'],do_sample=False,use_cache=True,
        use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
   generated=generated[:,inputs.input_ids.shape[1]:].tolist();del inputs
   for item,row,ids in zip(batch,raw,generated):
    folder=work/'conditions'/item['key'];q=folder/'qwen';q.mkdir(parents=True,exist_ok=True)
    attempts=[dict(token_ids=ids,max_new_tokens=p['planner_max_new_tokens'])]
    if eos not in ids:
     single=collator.source_inputs([row]).to(device,dtype=torch.bfloat16);runtime.thinker.rope_deltas=None
     with torch.inference_mode():
      gen=runtime.thinker.generate(**single,max_new_tokens=p['planner_retry_tokens'],do_sample=False,use_cache=True,
        use_audio_in_video=False,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id)
     ids=gen[0,single.input_ids.shape[1]:].tolist();del single,gen
     attempts.append(dict(token_ids=ids,max_new_tokens=p['planner_retry_tokens']))
    truncated=eos not in ids
    if truncated:
     save(q/'truncation.json',dict(reason='Planner hit token limit after native retry',attempts=attempts,kept_raw_output=True,synthetic_eos=False))
    ids=ids[:ids.index(eos)+1] if not truncated else ids
    text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
    if not text or '<|' in text or '|>' in text:raise ValueError('Invalid generated planning text')
    prompt=caption_prompt(text)
    with torch.inference_mode():payload=extract_planned_input(d,observation(item,folder,row['audio']),prompt,runtime,processor,32768)
    atomic_torch(q/'hidden.pt',payload)
    save(q/'plan.json',dict(description=text,prompt=prompt,source=p['planner'],adapter=adapter,
       system=SYSTEM,user=USER.format(duration=row['duration']),attempts=attempts,target_description_access=False,
       planner_truncated=truncated,finish_reason='length' if truncated else 'eos'))
    complete(q,[q/'hidden.pt',q/'plan.json'],source_only=True)
    progress(work,'cache',rank,part='qwen',sample=item['key'],done=start+batch.index(item)+1,total=len(qtodo))
   del raw,batch,generated
  del runtime,processor,collator
  if adapter:del peft,reader
  unload()
 todo=[x for x in items if not ready(work/'conditions'/x['key']/'text')]
 negative=work/'negative'/f'{rank:02d}'
 if todo or not ready(negative):
  from omni_av.ltx_runtime import NativeTextConditioner
  encoder=NativeTextConditioner(cfg,device)
  if not ready(negative):
   negative.mkdir(parents=True,exist_ok=True)
   atomic_torch(negative/'payload.pt',cpu(encoder.encode(FixedPrompt(''))));complete(negative,[negative/'payload.pt'])
  for start in range(0,len(todo),p['text_batch']):
   batch=todo[start:start+p['text_batch']]
   prompts=[read(work/'conditions'/x['key']/'qwen/plan.json')['prompt'] for x in batch]
   lengths=[len(encoder.encoder.tokenizer.tokenizer(x.strip(),truncation=False)['input_ids']) for x in prompts]
   with torch.inference_mode():
    values=encoder.encode_batch([FixedPrompt(x) for x in prompts]) if max(lengths)<=1024 else [encode_text(encoder,x,8192)[0] for x in prompts]
   for item,text in zip(batch,values):
    folder=work/'conditions'/item['key']/'text';folder.mkdir(parents=True,exist_ok=True)
    atomic_torch(folder/'payload.pt',cpu(text));complete(folder,[folder/'payload.pt'])
   del values
   progress(work,'cache',rank,part='gemma',done=min(start+p['text_batch'],len(todo)),total=len(todo))
  encoder.close();del encoder;unload()
