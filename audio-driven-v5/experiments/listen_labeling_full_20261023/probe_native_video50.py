import concurrent.futures,json,sys,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_labeling.pipeline import branch_inputs
from listen_labeling.client import prompt
from listen_labeling.guided import response_format
from listen_labeling.schema import parse,validate_response
from p2_listen.contracts import atomic_json,digest
out=Path(__file__).resolve().parent/'native_video50';out.mkdir(exist_ok=True)
rows=json.loads((out.parent/'comparison_best50'/'inputs.json').read_text())['samples']
media_cfg=dict(fps=8,max_frames=128,max_edge=768,max_audio_seconds=16)
strict='''Additional head-nod verification: A positive requires clear pitch rotation with a downward-to-upward recovery or upward-to-downward return. Sustained looking down, turning left/right, lateral body translation, camera movement, blinking, smiling, and speech-related jaw movement are not sufficient. Verify an ordered start, pitch extremum, and recovery using the actual frame timestamps; describe those observations and timestamps in description. If sampling, cropping or occlusion prevents verifying a cycle, use unknown instead of a confident positive. Do not stretch an isolated nod over the full clip. Only combine repeated cycles if visibly continuous, and count the cycles consistently. Never infer nodding just because this is a listening task.'''

def task(row,content,meta,native,tight,stage):
 label=stage;text=prompt('visual',row['duration_s'],stage)
 if tight:text+='\n'+strict
 parts=content;options={}
 if native:
  parts=[dict(type='video_url',video_url=dict(url='data:video/jpeg;base64,'+','.join(p['image_url']['url'].split(',',1)[1] for p in content if p['type']=='image_url')))]
  options=dict(media_io_kwargs=dict(video=dict(num_frames=-1,fps=8,frame_times_s=[f['time_s'] for f in meta['video']['frames']],clip_duration_s=row['duration_s'])),mm_processor_kwargs=dict(do_sample_frames=False))
 payload=dict(model='JoyAI-VL-Interaction-listen-gpu2',messages=[dict(role='user',content=[dict(type='text',text=text),*parts])],temperature=0,max_tokens=1536,chat_template_kwargs=dict(enable_thinking=False),response_format=response_format('visual',row['duration_s'],stage),**options)
 result=dict(sample_id=row['sample_id'],variant=label,input_images=len(meta['video']['frames']),input_frame_meta=meta,prompt=text,request_options=options,request_sha256=digest(payload));began=time.monotonic()
 try:
  response=requests.post('http://127.0.0.1:18262/v1/chat/completions',json=payload,timeout=300);result.update(status_code=response.status_code,body=response.text);response.raise_for_status();b=response.json();choice=b['choices'][0]
  if choice['finish_reason']!='stop':raise ValueError('truncated')
  result['parsed']=validate_response(parse(choice['message']['content']),'visual',row['duration_s'],stage);result['usage']=b['usage'];result['status']='schema_pass'
 except Exception as e:result.update(status='failed',error=str(e))
 result['seconds']=time.monotonic()-began;atomic_json(out/(row['sample_id']+'_'+label+'.json'),result)
 print(json.dumps({k:result.get(k) for k in ['sample_id','variant','status','error','seconds','usage','parsed']},ensure_ascii=False),flush=True)
prepared=[]
for row in rows:
 content,meta=branch_inputs(row,'visual',media_cfg);prepared.append((row,content,meta))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(task,*data,True,False,stage) for data in prepared for stage in ['discovery','audit']]
 for future in futures:future.result()
