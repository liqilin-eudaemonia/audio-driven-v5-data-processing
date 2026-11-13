"""Independent listener audio recheck, preserving every raw response."""
import argparse,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
PROMPT='''Describe every audible listener backchannel or vocal response with its start and end time in this audio clip. Distinguish spoken words from laughter and other nonverbal sounds. Keep repeated spoken responses as separate occurrences; keep a multiword response such as "uh huh" together. Use an open factual description, and the spoken text when intelligible; for nonverbal sounds text is empty. Do not turn laughter syllables into spoken words. Times are relative to the beginning of this clip. Return {"events":[{"start":number,"end":number,"description":string,"text":string,"kind":"speech"|"nonverbal"}]}.'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run=Path(a.run);out=run/'omni';out.mkdir(exist_ok=True)
 import numpy as np,soundfile as sf
 from scipy.signal import resample_poly
 from transformers import AutoProcessor
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 model='/private/ckpts/Qwen3-Omni/Instruct';processor=AutoProcessor.from_pretrained(model,trust_remote_code=True,local_files_only=True)
 llm=LLM(model=model,trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=.8,max_model_len=32768,max_num_seqs=32,max_num_batched_tokens=16384,limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,enforce_eager=False)
 jobs=json.loads((run/'omni_jobs.json').read_text())[a.shard::a.shards]
 for first in range(0,len(jobs),32):
  batch=[j for j in jobs[first:first+32] if not (out/(j['job_id']+'.json')).exists()];payloads=[];params=[]
  for j in batch:
   with sf.SoundFile(j['audio']) as f:
    rate=f.samplerate;f.seek(round(j['start']*rate));audio=f.read(round(j['end']*rate)-round(j['start']*rate),dtype='float32')
   audio=resample_poly(audio,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
   msg=[dict(type='text',text=PROMPT+f"\nClip duration: {j['duration_s']:.6f} seconds."),dict(type='audio',audio='listener://independent',audio_url='listener://independent')]
   prompt=processor.apply_chat_template([dict(role='user',content=msg)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   schema={'type':'object','additionalProperties':False,'required':['events'],'properties':{'events':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['start','end','description','text','kind'],'properties':{'start':{'type':'number','minimum':0,'maximum':round(j['duration_s'],6)},'end':{'type':'number','minimum':0,'maximum':round(j['duration_s'],6)},'description':{'type':'string'},'text':{'type':'string'},'kind':{'type':'string','enum':['speech','nonverbal']}}}}}}
   payloads.append(dict(prompt=prompt,multi_modal_data={'audio':[audio]},mm_processor_kwargs={'use_audio_in_video':False}));params.append(SamplingParams(temperature=0,max_tokens=4096,structured_outputs=StructuredOutputsParams(json=schema)))
  if not batch:continue
  start=time.time();outputs=llm.generate(payloads,params,use_tqdm=False)
  for j,o in zip(batch,outputs):
   item=o.outputs[0];record=dict(job=j,prompt=PROMPT,raw=item.text,finish_reason=item.finish_reason,seconds=time.time()-start,model=model,human_feedback_input=False)
   try:record['parsed']=json.loads(item.text)
   except ValueError:record['parsed']=None
   (out/(j['job_id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2));print(j['job_id'],flush=True)
 (run/f'omni-{a.shard}.complete.json').write_text(json.dumps(dict(jobs=len(jobs))))
if __name__=='__main__':main()
