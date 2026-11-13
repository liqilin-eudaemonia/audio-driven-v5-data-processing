"""Publish the same-24 comparison without changing prior runs or source media."""
import base64
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/p2_planning_sft_filtered_compare24_20261102'
SOURCE_MEDIA=ROOT/'data_previews/listen_benchmark_review/j_face224_training_20261101/media'


def mix_original(sample_id, number, destinations):
    """Listener video plus purified listener audio and speaker audio, sample-aligned."""
    src=SOURCE_MEDIA/sample_id
    video, listener, speaker=src/'listener.mp4', src/'listener.f32', src/'speaker.f32'
    for path in (video, listener, speaker):
        if not path.is_file():
            raise FileNotFoundError(path)
    name=f'sample-{number:02d}.mp4'
    target=destinations[0]/name
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg','-y','-v','error','-i',str(video),
        '-f','f32le','-ar','48000','-ac','1','-i',str(listener),
        '-f','f32le','-ar','48000','-ac','1','-i',str(speaker),
        '-filter_complex','[1:a]volume=0.5[l];[2:a]volume=0.5[s];[l][s]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]',
        '-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-b:a','160k','-movflags','+faststart',str(target)
    ], check=True)
    if target.stat().st_size<1000:
        raise RuntimeError(f'mixed video is empty: {target}')
    for folder in destinations[1:]:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, folder/name)
    return name


def main():
    read=lambda p:json.loads(p.read_text())
    data=read(OUT/'assessment.json');protocol=read(OUT/'protocol.json')
    requests=read(Path(protocol['source_requests']))
    metadata={x['id']:x for x in read(ROOT/'data_previews/listen_benchmark_review/j_face224_training_20261101/data.json')['samples']}
    assert len(data['rows'])==24 and data['baseline_replay_token_exact']==24
    esc=html.escape
    def embed(path,mime):return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()
    public=ROOT/'data_previews/listen_benchmark_review/planning_sft_filtered_compare24_20261102'
    media_dirs=[OUT/'media', public/'media']
    cards=[]
    for r in sorted(data['rows'],key=lambda x:(x['original_split']!='dev',x['number'])):
        q=requests[r['number']-1];assert q['sample_id']==r['sample_id']
        m=metadata[r['sample_id']];membership=r['latest_membership']
        if r['original_split']=='dev':exposure='原dev：未参与新旧训练的梯度更新'
        elif membership and membership['split']=='train':exposure='该样本ID进入最新训练集'
        else:exposure='该样本ID未进入最新训练集；原属train，不能视为独立留出样本'
        columns=''.join('<div class="'+('latest' if k=='latest' else '')+'"><h3>'+label+'</h3><p>'+esc(r['texts'][k])+'</p></div>' for k,label in [('base','未微调 · 相同prompt'),('epoch2','旧500h LoRA · 两轮'),('latest','新1,155h LoRA · 一轮 / 2697步')])
        reference=f'''<div class="reference"><h3>Speaker 音频 ASR</h3><p>{esc(str(m['speaker_text']))}</p><h3>原数据自然语言标注</h3><p>{esc(str(m['caption']))}</p><p class="note">这两项都来自原数据，没有输入规划模型。标注是一种已有描述，不是唯一合理回应。</p></div>'''
        hidden_reference=f'''<details><summary>已有speaker转写与listener弱标注（均未输入规划模型）</summary>{reference}</details>'''
        shown_reference=reference if r['original_split']=='dev' else hidden_reference
        if r['original_split']=='dev':
            mixed=mix_original(r['sample_id'], r['number'], media_dirs)
            playback=f'''<div class="original"><h3>原始片段</h3><video controls preload="metadata" src="media/{mixed}"></video><p>listener 画面，音轨是净化后的 listener 与 speaker 按同一起点叠加。净化音轨只保留官方词、句和 VAD 的并集，其余为静音。</p></div>'''
        else:
            playback=''
        cards.append(f'''<article data-split="{r['original_split']}" data-id="{esc(r['sample_id'])}" id="sample-{r['number']}">
<h2>#{r['number']:02d} · {r['original_split']} · {r['duration']:.3f} 秒</h2><small>{esc(r['sample_id'])}</small>
<p class="exposure">{exposure}</p><div class="sample"><aside><img alt="模型实际输入的listener最终裁框首帧" src="{embed(q['reference'],'image/png')}"><p>规划输入：listener 首帧＋下方 speaker 音频</p><audio controls preload="none" src="{embed(q['speaker_audio'],'audio/wav')}"></audio><p class="note">{esc(r['note'])}</p></aside>
<section>{playback}{shown_reference}<div class="columns">{columns}</div><details><summary>历史旧500h LoRA一轮结果</summary><p>{esc(r['texts']['epoch1'])}</p></details></section></div></article>''')
    table=[]
    for label,key in [('正常结束','eos_count'),('平均英文词数','mean_words'),('明确安排发声','explicit_vocal'),('明确安排静音','explicit_silent'),('发声状态未充分说明','unspecified_vocal')]:
        table.append('<tr><td>'+label+'</td>'+''.join('<td>'+str(data['summary'][k][key])+'</td>' for k in ['base','epoch1','epoch2','latest'])+'</tr>')
    table.append('<tr><td>引用不存在的补充信息</td>'+''.join('<td>'+str(len(data['summary'][k]['supplementary_evidence_numbers']))+'</td>' for k in ['base','epoch1','epoch2','latest'])+'</tr>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最新LoRA · 同24条回应规划对比</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1800px;margin:auto;padding:24px;background:#f3f4f6;color:#202124}h1{font-size:26px}article{background:white;border:1px solid #ddd;border-radius:10px;padding:22px;margin:24px 0;scroll-margin-top:100px}small{overflow-wrap:anywhere}h2,h3{margin:6px 0}.sample{display:grid;grid-template-columns:240px minmax(0,1fr);gap:24px}aside img,aside video{width:230px;max-width:100%;height:auto;background:#111}.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.columns>div{border-top:3px solid #8190a8}.columns>.latest{border-color:#167347;background:#f4faf6;padding:0 10px}p{white-space:pre-wrap}.note{background:#edf2ff;padding:12px}.reference{background:#fff8e8;border:1px solid #e4d3a2;border-radius:8px;padding:12px 16px;margin:0 0 16px}.reference h3{font-size:15px}.reference .note{background:transparent;padding:0;color:#5c5346;font-size:14px}.original{margin:0 0 16px}.original video{width:min(512px,100%);background:#111;display:block}.original p{font-size:13px;color:#444}.exposure{color:#555;font-size:14px}details{border-top:1px solid #ddd;padding:12px 0}summary{cursor:pointer}nav{position:sticky;top:0;background:#f3f4f6;padding:12px;z-index:1;border-bottom:1px solid #bbb}button,input{padding:8px 12px;margin:4px;max-width:95%;font:inherit}audio{width:100%;margin-top:12px}pre{white-space:pre-wrap}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccc;padding:7px 16px;text-align:left}@media(max-width:1200px){.columns{grid-template-columns:1fr}.sample{grid-template-columns:210px minmax(0,1fr)}}@media(max-width:650px){.sample{grid-template-columns:1fr}body{padding:12px}table{font-size:12px}td,th{padding:5px}}</style>
<h1>最新LoRA：与未微调、旧LoRA使用同24条输入</h1>
<p><b>明确改善：不再虚构“补充信息”依据，发声状态交代更完整。</b>但18/24条仍安排一次点头，描述模板较集中；不能据此断言对话理解或最终音视频生成已改善。</p>
<p>相同首帧、speaker波形、system/user prompt、batch24、greedy、max_new_tokens512。最新权重完成2697步；本次基线与历史基线24/24条token完全一致。三列比较主结果，旧一轮可展开查看。</p>
<p>原抽样为20条train＋4条dev；最新训练只包含其中12个样本ID。另8条原train虽未按本ID训练，源会话未必独立。4条原dev均未参与梯度更新，优先展示。此处比较规划文本，未生成LTX视频；统计不是准确率。</p>
<table><thead><tr><th>描述性统计</th><th>未微调</th><th>旧一轮</th><th>旧两轮</th><th>最新一轮</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table>
<p>优先复核：#1/#2/#4/#17 的虚构证据是否去除；#23 发声说明是否完整；#24 从“Yeah”改为笑声是否适合上下文。发声更多不自动等于更好。左侧单独音轨是规划输入的 speaker。#21–#24 另有原始叠加视频：listener 画面＋净化 listener 音轨＋speaker。</p>
<details><summary>统一system / user prompt</summary><pre>'''+esc(protocol['system']+'\n\n'+protocol['user'])+'''</pre></details>
<nav><button onclick="choose('all')">全部24条</button><button onclick="choose('dev')">原dev 4条</button><button onclick="choose('train')">原train 20条</button><input id="search" aria-label="搜索样本编号" placeholder="搜索样本ID" oninput="filter()"></nav>
'''+''.join(cards)+'''<script>let split='all';function choose(s){split=s;filter()}function filter(){let q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('article').forEach(x=>x.hidden=(split!=='all'&&x.dataset.split!==split)||!x.dataset.id.toLowerCase().includes(q))}</script></html>'''
    (OUT/'review.html').write_text(doc)
    public.mkdir(exist_ok=True)
    (public/'index.html').write_text(doc)
    flat=[dict(number=r['number'],sample_id=r['sample_id'],original_split=r['original_split'],latest_in_train=bool(r['latest_membership'] and r['latest_membership']['split']=='train'),**r['texts'],note=r['note']) for r in data['rows']]
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({'samples':len(cards),'output':str(public/'index.html'),'portable':str(OUT/'review.html')},ensure_ascii=False))


if __name__=='__main__':main()
