"""Paired visual-prompt pilot; never modifies production annotations."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
RUN=Path(__file__).resolve().parent
from listen_benchmark.prompts import annotation_prompt
def write(p,r):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def observation_prompt(duration):
 return f'''Watch this {duration:.9g}-second listener clip. Describe all directly visible movements and facial-expression changes you recognize.
For each occurrence, give its start time, end time (seconds from clip start), and a brief description of what visibly changes.
Describe physical observations only, without interpreting their psychological or communicative meaning.
Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).'''
def prepare():
 previous=ROOT/'experiments/listen_visual_observable50_20261029'
 assert not (RUN/'samples.json').exists(), 'Prepared inputs already exist'
 import shutil
 shutil.copyfile(previous/'samples.json',RUN/'samples.json')
 protocol=json.load(open(previous/'protocol.json'))
 protocol.update(arms=['baseline','observation_v1'],previous_experiment=str(previous),prompt_template=observation_prompt(6),sample_manifest_sha256=hashlib.sha256((RUN/'samples.json').read_bytes()).hexdigest(),change='Replace backchannel interpretation task with physical observations; specify continuous-event grouping',addition=None)
 protocol['runtime']['baseline_and_observable_same_batch']=True
 write(RUN/'protocol.json',protocol)
 print('Prepared identical 50 samples',flush=True)
def worker(rank,shards,memory):
 import torch
 from transformers import AutoProcessor
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 from listen_benchmark.full_raw import prepare as media_prepare,views
 from listen_benchmark.windows import slice_media
 from listen_benchmark.media import timestamped_images
 from listen_benchmark.simple_timeline import response_schema,validate_timeline
 torch.set_num_threads(2)
 model='/private/ckpts/Qwen3-VL-8B-Instruct'
 processor=AutoProcessor.from_pretrained(model,trust_remote_code=True,local_files_only=True)
 llm=LLM(model=model,dtype='bfloat16',gpu_memory_utilization=memory,max_model_len=49152,max_num_seqs=2,max_num_batched_tokens=4096,enable_chunked_prefill=True,limit_mm_per_prompt={'image':128,'video':0},skip_mm_profiling=True,enforce_eager=True,seed=20261029)
 done=0
 for row in json.load(open(RUN/'samples.json'))[rank::shards]:
  sid=row['sample_id'];path=RUN/'results'/f'{sid}.json'
  if path.exists():continue
  start=time.time();whole=media_prepare(row['source'],'visual');records=[]
  for name,a,b in views(whole['duration_s']):
   media=whole if name=='full' else slice_media(whole,a,b,8)
   base=annotation_prompt(dict(branch='visual',prompt_variant='backchannel_visual_simple_v2'),media['duration_s'],'discovery')
   prompts=[base,observation_prompt(media['duration_s'])];inputs=[];params=[]
   for prompt in prompts:
    parts=[dict(type='text',text=prompt)]+timestamped_images(media)
    rendered=processor.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs.append(dict(prompt=rendered,multi_modal_data={'image':media['images']}))
    params.append(SamplingParams(temperature=0,max_tokens=4096,structured_outputs=StructuredOutputsParams(json=response_schema(media['duration_s']))))
   responses=llm.generate(inputs,params,use_tqdm=False)
   for arm,prompt,response in zip(['baseline','observation_v1'],prompts,responses):
    o=response.outputs[0];rec=dict(arm=arm,view=name,start=a,end=b,prompt=prompt,raw=o.text,finish_reason=o.finish_reason,completion_tokens=len(o.token_ids),media=media['meta'])
    try:
     value=json.loads(o.text);rec['parsed']=value;validate_timeline(value,media['duration_s'])
     if any(e['modality']!='visual' for e in value['events']):raise ValueError('Wrong modality')
     if o.finish_reason!='stop':raise ValueError('Truncated output')
     rec['status']='valid'
    except Exception as e:rec.update(status='invalid',error=str(e))
    records.append(rec)
   write(RUN/'partial'/f'{sid}.json',dict(sample_id=sid,records=records))
  write(path,dict(sample_id=sid,duration_s=whole['duration_s'],records=records,elapsed=time.time()-start))
  done+=1;write(RUN/'workers'/f'{rank}.json',dict(pid=os.getpid(),rank=rank,completed=done,last_sample=sid,time=time.time()))
  print(sid,'done',flush=True)
 write(RUN/'workers'/f'{rank}.complete.json',dict(completed=done,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--memory',type=float,default=.20);a=p.parse_args()
 if a.prepare:prepare()
 else:worker(a.rank,a.shards,a.memory)
