import sys,copy,math,shutil,concurrent.futures,time
from pathlib import Path
import soundfile as sf
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_glare_eval.common import artifact_valid
import build_p2_av_review as media
OLD=media.OUT;OUT=OLD/'autonomous_preview_20261028';OUT.mkdir(exist_ok=True);media.OUT=OUT;media.MEDIA=OUT/'media';media.MEDIA.mkdir(exist_ok=True)

def main():
 jobs=read(HERE/'preview_jobs.json');scope=read(HERE/'preview_scope.json');ids=set(scope['sample_ids'])
 old=next(d for d in read(OLD/'data.json')['datasets'] if d['id']=='test');samples=[];tasks=[];count=0
 for original in old['samples']:
  if original['id'] not in ids:continue
  s=copy.deepcopy(original);s['outputs']=[];s['index']=len(samples)+1
  for key in ['image','speaker','gt_audio','gt_video','metric_gt_audio']:
   if s.get(key):s[key]='../'+s[key]
  for j in jobs:
   if j['sample_id']!=s['id']:continue
   out=Path(j['output']);marker=out/'complete.json'
   if not marker.exists():continue
   c=read(marker);assert artifact_valid(out,c['contract'])
   meta=read(out/'completed.json');assert meta['plan_source']=='qwen_autonomous' and meta['target_annotation_access'] is False and meta['gt_media_access'] is False
   plan=read(out/'plan.json');prompt=(out/'prompt.txt').read_text();assert prompt==plan['prompt']
   wave,_=sf.read(out/'listener.wav');rms=float(np.sqrt(np.mean(wave*wave)))
   name='AV＋Event' if 'av_event' in j['arm'] else 'AV'
   entry=dict(arm=j['arm'],name=name,variant='guided',seed=j['seed'],video=media.link(out/'generated.mp4'),audio=media.link(out/'listener.wav'),prompt=prompt,rms=rms,guidance_language='English',metric_status='not_evaluated',metrics=dict(utmos=None,lse_d=None,lse_c=None,rms_dbfs=20*math.log10(max(rms,1e-15))),plan_record=media.link(out/'plan.json'))
   s['outputs'].append(entry);tasks.append((entry,out/'generated.mp4',out/'listener.wav',s['duration'],s['sources']['speaker']));count+=1
  samples.append(s)
 if count==0:raise RuntimeError('No completed preview videos yet')
 with concurrent.futures.ThreadPoolExecutor(4) as pool:
  fs={pool.submit(media.mux,v,a,d,sp):entry for entry,v,a,d,sp in tasks}
  for f in concurrent.futures.as_completed(fs):fs[f]['mixed_video']=f.result()
 atomic_json(OUT/'data.json',dict(schema='autonomous-planning-preview-v1',datasets=[dict(id='test',samples=samples)],generated_outputs=count,planned_outputs=12,samples=3,metrics_computed=False))
 shutil.copy2(OLD/'style.css',OUT/'style.css')
 html=(OLD/'index.html').read_text();start=html.find('<div class="review-note" id="reintegrated-batch">')
 if start>=0:
  end=html.index('</div>',start)+6;html=html[:start]+html[end:]
 html=html.replace('<h1>听见输入，对照回应</h1>',f'<h1>Qwen自主规划 · 3条视频预览</h1><p><a href="../">返回历史评测</a> · 已生成 {count}/12 个输出（AV / AV＋Event，seed137/251）。仅3/24条规划通过校验；是成功样本预览，不是完整评测结果。</p>')
 html=html.replace('<button data-dataset="train" aria-pressed="true">训练集24条</button>','').replace('独立test24条','独立test中3条预览')
 html=html.replace('<label>生成条件 <select id="variant">','<label style="display:none">生成条件 <select id="variant">')
 a=html.index('<section class="metrics-panel">');b=html.index('</section>',a)+len('</section>')
 html=html[:a]+'<section class="metrics-panel"><h2>预览范围与规划</h2><p>两组均为标注指导＋冻结Qwen训练的最终step1249权重。这次只用speaker音频、listener首帧和时长生成规划，同一规划同时进入Qwen第二次forward与Gemma。没有用目标标注生成回应。本页尚未计算评测指标。</p><p>通过校验的3条规划仅有简单点头／眨眼，没有规划发声；实际生成音轨保持原样，是否发声请以视频和独立音轨为准。</p></section>'+html[b:]
 html=html.replace('目标标注指导属于额外信息条件，不能与仅speaker／首帧输入的结果混为同一评测。所有原始输出保留。','自主规划小样本预览：3/24条规划有效，其余未通过；不能代表全test24表现。原始规划、视频与音轨全部保留。')
 html=html.replace('data-mode="listener" aria-pressed="true"','data-mode="listener" aria-pressed="false"').replace('data-mode="mixed" aria-pressed="false"','data-mode="mixed" aria-pressed="true"').replace('src="app.js"','src="app.js?v=speaker-mix-default"')
 (OUT/'index.html').write_text(html)
 js=(OLD/'app.js').read_text().replace("dataset:'train'","dataset:'test'").replace("dataset('train')","dataset('test')")
 # Keep the original tested media controls, but replace all oracle/status wording.
 js=js.replace("'英文标注指导 · 已完成结果'","'Qwen自主规划 · 视频预览'").replace("'标注指导'","'Qwen自主规划'")
 js=js.replace('UTMOS / SyncNet：评测正在补齐','本页仅视频预览，尚未计分')
 js=js.replace("$('#metric-scope').addEventListener('change',e=>{state.metricScope=e.target.value;renderMetrics(database.datasets.find(d=>d.id===state.dataset))});",'')
 js=js.replace("$('#glare-protocol').addEventListener('change',e=>{state.glareProtocol=e.target.value;renderMetrics(database.datasets.find(d=>d.id===state.dataset))});",'')
 js+='\nfunction renderMetrics(d) {}\n'
 js=js.replace('${o.prompt?`','${o.plan_record?`<p><a href="${esc(o.plan_record)}" target="_blank" rel="noopener">查看Qwen原始规划与结构化记录</a></p>`:\'\'}${o.prompt?`')
 js=js.replace("mode:'listener'","mode:'mixed'").replace("dataset('test');window.reviewReady=true","dataset('test');changeMode(state.mode);window.reviewReady=true")
 (OUT/'app.js').write_text(js)
 atomic_json(OUT/'publication.json',dict(at=time.time(),outputs=count,planned_outputs=12,samples=3,metrics_computed=False,url='http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/'))
 print(dict(outputs=count,url='http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/'),flush=True)
 if (HERE/'preview_metrics/comparison.json').exists():
  from publish_preview_metrics import publish
  publish()
if __name__=='__main__':main()
