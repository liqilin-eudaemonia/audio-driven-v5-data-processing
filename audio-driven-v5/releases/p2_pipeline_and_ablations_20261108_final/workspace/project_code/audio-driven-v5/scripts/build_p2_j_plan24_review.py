"""Build a self-contained review of unedited predictions and held-out references."""
import base64
import csv
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_j_omni3b_autonomous24_20261101'
REVIEW=ROOT/'data_previews/listen_benchmark_review/j_face224_training_20261101'


def data_uri(path,mime):
    return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def main():
    predictions=json.loads((WORK/'results.json').read_text())
    requests={r['sample_id']:r for r in json.loads((WORK/'requests.json').read_text())}
    references={r['id']:r for r in json.loads((REVIEW/'data.json').read_text())['samples']}
    assessments={r['number']:r for r in json.loads((WORK/'assessment.json').read_text())['samples']}
    cards=[];rows=[];escape=html.escape
    for p in predictions:
        sid=p['sample_id'];r=references[sid];q=requests[sid]
        rows.append(dict(number=p['number'],sample_id=sid,split=r['split'],duration_s=r['duration'],
            speaker_text_reference=r.get('speaker_text',''),prediction=p['generated_text'],
            listener_caption_reference=r['caption'],listener_reference_has_audio=not r['listener_audio']['full_silence'],
            ended_with_eos=p['ended_with_eos'],generated_tokens=p['generated_tokens'],
            review_note=assessments[p['number']]['note']))
        cards.append(f'''<article id="sample-{p['number']}"><h2>{p['number']:02d} · {escape(r['split'])} · {r['duration']:.3f}s</h2>
<small>{escape(sid)}</small><div class="row"><div><img alt="最终裁框参考首帧" src="{data_uri(q['reference'],'image/png')}">
<p>实际模型输入：speaker音频</p><audio controls preload="none" src="{data_uri(q['speaker_audio'],'audio/wav')}"></audio></div>
<div><h3>Qwen-Omni-3B原始规划</h3><p class="prediction">{escape(p['generated_text'])}</p>
<small>EOS：{p['ended_with_eos']} · {p['generated_tokens']} tokens</small>
<p><strong>初步检查：</strong>{escape(assessments[p['number']]['note'])}</p>
<details><summary>生成后对照：speaker原转写与listener标注（均未输入模型）</summary>
<h4>speaker转写，仅供辅助核对</h4><p>{escape(r.get('speaker_text',''))}</p>
<h4>listener已有弱标注，不是唯一合理回应</h4><p>{escape(r['caption'])}</p>
<p>参考listener音轨：{'全静音' if r['listener_audio']['full_silence'] else '非零音轨'}；标注框与最终框{'不同' if r['annotation_crop_differs_from_final'] else '相同'}。</p></details></div></div></article>''')
    with (WORK/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    protocol=json.loads((WORK/'protocol.json').read_text())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>未微调Qwen-Omni-3B：24条自主回应规划</title><style>
body{font:16px/1.65 system-ui;background:#f4f5f7;color:#17202a;max-width:1120px;margin:32px auto;padding:0 20px}
article{background:white;border:1px solid #dce1e7;border-radius:12px;padding:24px;margin:24px 0}.row{display:grid;grid-template-columns:280px 1fr;gap:28px;margin-top:18px}
img{width:100%;border-radius:8px}audio{width:100%}small{overflow-wrap:anywhere;color:#576574}h2,h3{margin:0 0 12px}.prediction{white-space:pre-wrap;font-size:18px}
details{background:#f0f3f6;border-radius:8px;padding:14px;margin-top:18px}summary{cursor:pointer}pre{white-space:pre-wrap}@media(max-width:700px){.row{grid-template-columns:1fr}img{max-width:280px}}
</style><h1>未微调Qwen-Omni-3B：24条自主回应规划</h1>
<p>原20条train＋4条dev；最终裁框首帧＋speaker音频＋固定system/user指令。没有输入真实listener描述、音轨、后续帧、事件或speaker转写。24条一次合批，greedy自由生成，无LoRA、无JSON约束、无重试挑选。</p>
<p>点击音频核对speaker语义与语气。原标注仅供展开对照；不同于原标注不等于规划错误。这里只评审文本计划，尚未生成listener音视频。</p>'''
    page+='<details><summary>查看完整固定prompt</summary><h3>System</h3><pre>'+escape(protocol['system'])+'</pre><h3>User</h3><pre>'+escape(protocol['user_template'])+'</pre></details>'
    page+=''.join(cards)+'''<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''
    (WORK/'review.html').write_text(page)
    print(json.dumps(dict(samples=len(rows),html_bytes=len(page.encode()),all_eos=all(p['ended_with_eos'] for p in predictions))))


if __name__=='__main__':main()
