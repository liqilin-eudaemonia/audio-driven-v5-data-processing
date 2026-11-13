"""Paired visual-prompt pilot; never modifies production annotations."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
RUN=Path(__file__).resolve().parent
from listen_benchmark.prompts import annotation_prompt
ADDITION='Describe only directly visible actions and expressions. Do not infer psychological states, intentions, motivations, or communicative meanings.'
def write(p,r):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def prepare():
 subset=ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027'
 targets=json.load(open(subset/'manifests/targets.json'));chosen=random.Random(20261029).sample(sorted(r['sample_id'] for r in targets),50)
 source=subset/'integrated_labels.jsonl';rows={}
 with source.open() as f:
  for line in f:
   r=json.loads(line)
   if r['sample_id'] in chosen:rows[r['sample_id']]=dict(source=r['source'],previous_events=r['events'],line_sha256=hashlib.sha256(line.encode()).hexdigest())
 assert len(rows)==50
 write(RUN/'samples.json',[dict(sample_id=s,**rows[s]) for s in chosen])
 write(RUN/'protocol.json',dict(seed=20261029,samples=50,subset=str(subset),model='/private/ckpts/Qwen3-VL-8B-Instruct',addition=ADDITION,temperature=0,max_tokens=4096,window_seconds=3,stride_seconds=1,fps=8,original_crop=True,arms=['baseline','observable'],production_modified=False,media_truth_available=False))
 print('Prepared 50 samples',flush=True)
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
   prompts=[base,base+'\n'+ADDITION];inputs=[];params=[]
   for prompt in prompts:
    parts=[dict(type='text',text=prompt)]+timestamped_images(media)
    rendered=processor.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs.append(dict(prompt=rendered,multi_modal_data={'image':media['images']}))
    params.append(SamplingParams(temperature=0,max_tokens=4096,structured_outputs=StructuredOutputsParams(json=response_schema(media['duration_s']))))
   responses=llm.generate(inputs,params,use_tqdm=False)
   for arm,prompt,response in zip(['baseline','observable'],prompts,responses):
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
