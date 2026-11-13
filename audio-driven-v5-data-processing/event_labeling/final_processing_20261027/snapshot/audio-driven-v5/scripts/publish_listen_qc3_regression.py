"""Expose development examples and their unchanged automatic outcomes for listening."""
import json,html
from pathlib import Path
import soundfile as sf
import preview_listen_v6 as preview
from p3_listen.common import read,write,sha
from screen_listen_qc_sample import read_wave
ROOT=preview.ROOT;OUT=ROOT/'data_previews/seamless_listen_qc3_review';EXP=ROOT/'experiments/p3_listen_qc3_review'
reg=read(EXP/'known_examples_regression.json')['rows']
qc1=ROOT/'data_previews/seamless_listen_qc_review';qc2=ROOT/'data_previews/seamless_listen_qc2_review'
lookup={r['sample_id']:(folder,r) for folder in [qc1,qc2] for r in read(folder/'samples.json')['samples']}
ab={r['sample_id']:r for r in read(ROOT/'data_previews/seamless_listen_v6_audio_ab/samples.json')['samples']}
inv=read(ROOT/'experiments/p3_listen_v6_audio_ab/source_inventory.json');folder=OUT/'regression_audio';folder.mkdir(exist_ok=True)
labels={'keep':'保留候选','review':'待复核','exclude':'建议排除'};hlabels={'reject_noise':'用户确认含噪','reject_speaker_bleed':'用户指出串音','retain_valid_listener_response':'用户确认正常回应'}
cards=[]
for i,r in enumerate(reg):
 sid=r['sample_id'];destination=folder/(sid+'.wav')
 if sid in lookup:
  source,row=lookup[sid];path=source/row['preview']/'listener.wav'
  if not destination.exists():destination.symlink_to(path)
 else:
  row=ab[sid];wave=read_wave(inv[row['listener_id']]['normalization']['path'],row['start'],row['duration_s']);sf.write(destination,wave,48000,subtype='PCM_24')
 cards.append('<section><h3>'+str(i+1)+'. '+html.escape(sid)+'</h3><p>'+hlabels[r['human_label']]+' → 自动：<b>'+labels[r['automatic_status']]+'</b></p><audio controls preload="none" src="regression_audio/'+html.escape(sid)+'.wav"></audio></section>')
(OUT/'known_examples.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QC3 已知例回归试听</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:auto;padding:20px;background:#f1f4f8;color:#172636}section{background:white;padding:16px;margin:12px 0;border:1px solid #dce4ed;border-radius:10px}h3{overflow-wrap:anywhere;font-size:15px}audio{width:100%}a{color:#076ba8}</style><h1>已知例回归：成功与漏放都保留</h1><p><a href="./">返回主页面</a></p><p>这些例子参与过方案讨论，不能用来报告独立准确率。这里显示自动结果，没有用人工标签替换它。全部播放既有音量归一化音轨，无新降噪。</p>'''+''.join(cards)+'</html>')
p=ROOT/'scripts/listen_qc3_review.html';s=p.read_text().replace('<p><a href="annotation.html">事件标注队列</a></p>','<p><a href="annotation.html">事件标注队列</a> · <a href="known_examples.html">已知例回归试听</a></p>');p.write_text(s);(OUT/'index.html').write_text(s)
write(EXP/'extras_manifest.json',dict(script_sha256=sha(Path(__file__)),regression_audio_files=len(cards),note='Publishing only; does not change frozen screening decisions.'))
print('Published regression examples',len(cards))
