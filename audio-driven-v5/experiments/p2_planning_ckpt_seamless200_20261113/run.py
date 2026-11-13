"""Seven frozen Qwen checkpoints, current fixed Seamless200, node11 GPU7."""
import importlib.util,json,os,socket,sys,time,traceback,hashlib
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
SOURCE=ROOT/'experiments/p2_planning_ckpt_audio64_20261113/run.py'
spec=importlib.util.spec_from_file_location('native_qwen_diagnostic',SOURCE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
native.WORK=WORK
read,save,sha=native.read,native.save,native.sha

def prepare():
 import numpy as np
 import soundfile as sf
 from PIL import Image
 current=read(ROOT/'evaluation/SEAMLESS_CURRENT.json')
 prepared=Path(current['prepared_200']['prepared'])
 refs=[json.loads(line) for line in (prepared/'references.jsonl').read_text().splitlines()]
 assert len(refs)==len({r['id'] for r in refs})==200
 assert sha(prepared/'references.jsonl')==read(prepared/'seal.json')['references_sha256']
 inputs=read(ROOT/'experiments/glare10618_seamless_test6624_20261113/protocol.json')
 assert inputs['source_version']==current['version']
 byid={s['id']:s for s in inputs['samples']}
 samples=[]
 for r in refs:
  s=byid[r['id']]
  assert r['gt_reference']['row_sha256']==s['evaluation_reference']['gt_reference']['row_sha256']
  assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
  audit=read(s['source_input_audit'])
  assert audit['source_only'] and not audit['target_description_in_model_input'] and not audit['listener_future_media_in_model_input']
  wave,rate=sf.read(s['audio'],dtype='float32')
  assert rate==16000 and wave.ndim==1 and len(wave)==round(r['duration_s']*16000) and np.isfinite(wave).all()
  folder=WORK/'media'/r['id'];folder.mkdir(parents=True,exist_ok=True)
  for name,src in [('first_frame.png',s['reference']),('original.wav',s['audio'])]:
   dst=folder/name
   if dst.exists():assert dst.resolve()==Path(src).resolve()
   else:dst.symlink_to(src)
  samples.append(dict(id=r['id'],duration=r['duration_s'],split='test',reference_text=r['latest_annotation']['caption'],reference_human_ground_truth=False,reference_annotation=r['latest_annotation'],core48=r['core48'],input_audit=audit,files={name:dict(path=str((folder/name).relative_to(WORK)),sha256=sha(folder/name)) for name in ['first_frame.png','original.wav']}))
 save(WORK/'samples.json',samples)
 save(WORK/'protocol.json',dict(schema='qwen-seven-checkpoints-fixed-seamless200-v1',at=time.time(),arms=native.arms(),cases=200,total_outputs=1400,conditions=['original'],source_version=current['version'],references=str(prepared/'references.jsonl'),references_sha256=sha(prepared/'references.jsonl'),samples_sha256=sha(WORK/'samples.json'),physical_gpu=7,model=read(native.CFG)['model'],config=str(native.CFG),generation=dict(batch_size=16,do_sample=False,dtype='bfloat16',attn_implementation='sdpa',max_new_tokens=512,retry_max_new_tokens=1024,max_pixels=read(native.CFG)['max_pixels']),native_entry=str(SOURCE),native_entry_sha256=sha(SOURCE),scope='Qwen planning only; same real speaker audio and listener first frame; no listener future or target text in prompt',reference_policy='Automatic labels for comparison, not human ground truth; this selected test subset is diagnostic and not an unbiased checkpoint-selection dev set'))
 save(WORK/'PREPARED.json',dict(success=True,at=time.time(),samples=200,outputs=1400,source_version=current['version'],reused_verified_input_pairs=200))

def report():
 import collections,re
 p=read(WORK/'protocol.json');samples=read(WORK/'samples.json');stats=[];views=[]
 for arm in p['arms']:
  rows=[read(WORK/'results'/arm['name']/s['id']/'original.json') for s in samples]
  assert all(x['arm']==arm['name'] and not x['target_in_prompt'] for x in rows)
  texts=[x['text'] for x in rows]
  stats.append(dict(arm=arm['name'],count=len(rows),nonempty=sum(bool(t) for t in texts),ended_with_eos=sum(x['ended_with_eos'] for x in rows),retries=sum(len(x['attempts'])>1 for x in rows),mean_words=sum(len(t.split()) for t in texts)/len(texts),unique_texts=len(set(texts))))
 for s in samples:
  views.append(dict(sample=s,plans=[dict(arm=a['name'],text=read(WORK/'results'/a['name']/s['id']/'original.json')['text']) for a in p['arms']]))
 save(WORK/'RESULTS.json',dict(at=time.time(),stats=stats,scope='Generation coverage and descriptive statistics; reference behavior/timing accuracy requires review, no semantic ranking inferred'))
 payload=json.dumps(views,ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Seamless200 七组Qwen规划</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:24px auto;padding:0 16px;background:#f4f6fa;color:#182233}article{padding:18px;background:white;margin:16px 0;border-radius:10px}p{white-space:pre-wrap}img{width:160px}select{max-width:100%}select,button{font:inherit;padding:8px}.plans{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.plans{grid-template-columns:1fr}}audio{max-width:100%}</style><h1>Seamless固定200：七组Qwen规划</h1><p>同一真实speaker音频＋listener首帧；原模型和SFT500／1000／1500／2000／2500／2752。参考为现有自动标注，不输入模型。重点检查反应、发声及时间是否与标注一致；本页尚无人工准确率评分。</p><select id="choose"></select><button id="prev">上一条</button><button id="next">下一条</button><div id="view"></div><script>const data=PAYLOAD;let idx=0;const choose=document.getElementById('choose'),view=document.getElementById('view');data.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${d.sample.id}`;choose.append(o)});function art(title,text){const a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p');h.textContent=title;p.textContent=text;a.append(h,p);return a}function draw(){choose.value=idx;view.replaceChildren();const d=data[idx],s=d.sample,h=art(`${idx+1}/200 · ${s.duration.toFixed(3)}秒`,s.id);let im=document.createElement('img');im.src=s.files['first_frame.png'].path;h.append(im);let au=document.createElement('audio');au.controls=true;au.preload='none';au.src=s.files['original.wav'].path;h.append(au);view.append(h,art('现有标注（参考）',s.reference_text));let grid=document.createElement('div');grid.className='plans';d.plans.forEach(p=>grid.append(art(p.arm,p.text)));view.append(grid)}choose.onchange=()=>{idx=Number(choose.value);draw()};document.getElementById('prev').onclick=()=>{idx=(idx+data.length-1)%data.length;draw()};document.getElementById('next').onclick=()=>{idx=(idx+1)%data.length;draw()};draw();</script>'''.replace('PAYLOAD',payload)
 (WORK/'index.html').write_text(page)
 save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),samples=200,outputs=1400,all_nonempty=all(s['nonempty']==200 for s in stats),all_eos=all(s['ended_with_eos']==200 for s in stats),semantic_review_pending=True,physical_gpu=7,results_sha256=sha(WORK/'RESULTS.json')))

if __name__=='__main__':
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
  import subprocess
  uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  assert uuid=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
  save(WORK/'DEVICE.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),physical_gpu=7,uuid=uuid))
  if not (WORK/'PREPARED.json').exists():prepare()
  p=read(WORK/'protocol.json');assert sha(SOURCE)==p['native_entry_sha256'] and sha(WORK/'samples.json')==p['samples_sha256']
  native.generate(0,slots=1)
  report()
 except BaseException as e:
  save(WORK/'FAILURE.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
