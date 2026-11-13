"""Build a portable review of the frozen 24-sample planning comparison."""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/p2_planning_sft_compare24_20261101'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


NOTES = [
    '两轮都保持微笑、静音，并引用并不存在于输入中的补充信息；第二轮未解决此问题。',
    '两轮逐字相同，都提到补充信息里的点头；这类证据没有提供给规划模型。',
    '第一轮提出点头但没有交代发声；第二轮明确静音，动作更少。不能据此认定回应更合适。',
    '第一轮静止、静音；第二轮新增“补充信息说明点头”的无依据表述。',
    '两轮逐字相同：向上看、轻微闭嘴微笑、静音。',
    '都描述转移目光、末尾点头和静音；第二轮多写耳机外观，实质变化小。',
    '两轮都花较多篇幅描述衣着、背景及姿势，静音；对说话内容的回应不鲜明。',
    '第二轮删掉补充信息表述，加入末尾点头；仍静音，属于局部改善。',
    '第二轮缩短描述并调整点头时机，仍是抬头看、点头、静音。',
    '第一轮写嘴部像在说话但无声音；第二轮改为点头、无发声，描述更明确。',
    '第二轮加入手遮嘴再放下的动作；动作更多并不等于更符合上下文。',
    '两轮逐字相同：保持姿势、表情中性、静音。',
    '两轮逐字相同：表情中性、约三秒点头、静音。',
    '点头从约六秒改到约五秒，增加眨眼；仍静音，没有明确质量提升。',
    '两轮逐字相同：表情中性、约三秒点头、静音。',
    '第二轮把躺姿改成坐姿，并增加点头和低声轻笑；变化较大，但缺乏证据判断回应是否更好。',
    '第二轮新增“补充信息说明没有点头”，属于输入证据幻觉。',
    '第二轮去掉补充信息表述，增加眨眼和视线变化；局部更干净，仍静音。',
    '第二轮增加短暂向右看；仍以静止和静音为主。',
    '两轮都规划点头、视线变化和静音，主要是动作细节调整。',
    'dev：两轮都有约1.5秒点头、静音；第二轮减少具体的转头叙述，没有明显优劣。',
    'dev：两轮逐字相同，眨眼、抿嘴、末尾点头、静音。',
    'dev：第一轮有两次点头及末尾轻微气声；第二轮只有微笑和一次点头，漏掉是否发声的交代。',
    'dev：第一轮仅说没有清晰语音，未明确非语言声音；第二轮明确给出短回应“Yeah”，规划更具体，但不等于唯一正确答案。',
]


def main():
    names = ['base', 'epoch1', 'epoch2', 'epoch2_silence']
    runs = {name: read(OUT / f'{name}.json') for name in names}
    requests = read(ROOT / 'experiments/p2_j_omni3b_autonomous24_20261101/requests.json')
    metadata = {r['id']: r for r in read(ROOT / 'data_previews/listen_benchmark_review/j_face224_training_20261101/data.json')['samples']}
    ids = [r['sample_id'] for r in requests]
    manifest = read(ROOT / 'experiments/p2_j_train500h_cache_20261101/manifest.json')
    train_ids = set(manifest['splits']['train'])
    for run in runs.values():
        assert [r['sample_id'] for r in run['results']] == ids
        assert len(run['results']) == 24 and all(r['ended_with_eos'] for r in run['results'])
        assert run['system'] == runs['base']['system'] and run['user'] == runs['base']['user']
        assert run['target_in_prompt'] is False
    assert runs['epoch2_silence']['audio_ablation'] == 'silence'
    assert all((r['sample_id'] in train_ids) == (r['split'] == 'train') for r in requests)
    summaries = {}
    for name in names:
        rows = runs[name]['results']
        summaries[name] = dict(count=len(rows), eos_count=sum(r['ended_with_eos'] for r in rows),
            mean_words=round(mean(len(r['text'].split()) for r in rows), 1),
            supplementary_evidence_numbers=[i+1 for i, r in enumerate(rows) if 'supplement' in r['text'].lower()])
    rows = []
    for i, request in enumerate(requests):
        number = i+1
        texts = {k: runs[k]['results'][i]['text'] for k in names}
        voice1 = 'vocal' if number == 23 else ('unspecified' if number in (3, 24) else 'silent')
        voice2 = 'vocal' if number in (16, 24) else ('unspecified' if number == 23 else 'silent')
        rows.append(dict(number=number, sample_id=ids[i], split=request['split'], **texts,
            epoch1_vocal_status=voice1, epoch2_vocal_status=voice2,
            epoch1_epoch2_exact_match=texts['epoch1'] == texts['epoch2'],
            silence_exact_match=texts['epoch2'] == texts['epoch2_silence'], note=NOTES[i]))
    assessment = dict(summary=summaries, rows=rows,
        basis='Assistant qualitative inspection of generated text, reference images and existing speaker transcripts; no blinded human preference study or AV generation.',
        limitation='20 seen training samples and 4 held-out dev samples; captions are weak labels and responses are one-to-many.',
        epoch1_epoch2_exact_numbers=[r['number'] for r in rows if r['epoch1_epoch2_exact_match']],
        silence_exact_numbers=[r['number'] for r in rows if r['silence_exact_match']])
    save(OUT / 'assessment.json', assessment)
    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = read(OUT / 'protocol.json')
    protocol['silence_ablation'] = dict(adapter='epoch2', batch_size=24, max_new_tokens=512,
        do_sample=False, replacement='np.zeros_like(source_audio), preserving duration',
        generator_sha256=hashlib.sha256((ROOT / 'scripts/p2_planning_generate.py').read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256((OUT / 'epoch2_silence.json').read_bytes()).hexdigest())
    protocol['verified_sample_membership'] = dict(seen_train=20, held_out_dev=4)
    save(OUT / 'protocol.json', protocol)
    esc = html.escape

    def embed(path, mime):
        return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode()

    cards = []
    for r in sorted(rows, key=lambda r: (r['split'] != 'dev', r['number'])):
        req = requests[r['number']-1]
        meta = metadata[r['sample_id']]
        columns = ''.join(f'<div><h3>{label}</h3><p>{esc(r[key])}</p></div>' for key, label in
                          [('base', '未微调 · 相同prompt'), ('epoch1', 'LoRA 1轮'), ('epoch2', 'LoRA 2轮')])
        cards.append(f'''<article data-split="{r['split']}" id="sample-{r['number']}">
<h2>#{r['number']:02d} · {r['split']} · {req['duration_s']:.3f}s</h2>
<small>{esc(r['sample_id'])}</small>
<div class="input"><img alt="listener最终裁框参考首帧" src="{embed(req['reference'], 'image/png')}">
<div><b>实际输入：listener首帧＋speaker音频</b><br><audio controls preload="none" src="{embed(req['speaker_audio'], 'audio/wav')}"></audio>
<p>{'训练中见过的样本' if r['split'] == 'train' else '未参与梯度训练的dev样本'}</p>
<p class="note">{esc(r['note'])}</p></div></div>
<div class="columns">{columns}</div>
<details><summary>同一第二轮模型，speaker音频替换为等长静音（{'逐字相同' if r['silence_exact_match'] else '输出变化'}）</summary><p>{esc(r['epoch2_silence'])}</p></details>
<details><summary>已有speaker转写及弱标注（均未输入模型）</summary><h3>Speaker转写</h3><p>{esc(str(meta['speaker_text']))}</p><h3>原始listener弱标注</h3><p>{esc(str(meta['caption']))}</p><p>旧标注裁框与最终裁框是否不同：{meta['annotation_crop_differs_from_final']}。规划允许多种合理回应，不能把与标注不同直接判错。</p></details>
</article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>24条回应规划 · Base / LoRA 1轮 / LoRA 2轮</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1580px;margin:auto;padding:24px;background:#f3f4f6;color:#202124}h1{font-size:26px}article{background:white;border:1px solid #ddd;border-radius:10px;padding:22px;margin:24px 0;scroll-margin-top:70px}small{overflow-wrap:anywhere}h2,h3{margin:6px 0}.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.columns>div{border-top:3px solid #425f99}.columns p{white-space:pre-wrap}.input{display:flex;gap:24px;margin:18px 0}.input img{width:180px;height:180px;object-fit:contain;background:#eee}.note{background:#edf2ff;padding:12px}details{border-top:1px solid #ddd;padding:12px 0}summary{cursor:pointer}nav{position:sticky;top:0;background:#f3f4f6;padding:12px;z-index:1}button{padding:8px 16px;margin-right:8px;cursor:pointer}audio{max-width:100%;margin-top:12px}pre{white-space:pre-wrap}@media(max-width:850px){.columns{grid-template-columns:1fr}.input{flex-direction:column}}</style>
<h1>同24条样本：未微调 / LoRA 1轮 / LoRA 2轮</h1>
<p><b>描述完整性改善；第二轮未显示一致优于第一轮。</b>三组均使用当前SFT相同prompt，batch 24、greedy、max_new_tokens 512；全部正常结束。20条train在训练中见过，4条dev未参与梯度更新，优先展示dev。</p>
<p>两轮各4/24条引用输入中不存在的“补充信息”。静音替换使第二轮22/24条输出变化，只说明对音频输入敏感，不能证明语义理解和回应质量。此页比较规划文本，没有生成AV，也没有盲评。</p>
<details><summary>本次统一system / user prompt</summary><pre>''' + esc(runs['base']['system'] + '\n\n' + runs['base']['user']) + '''</pre></details>
<nav>显示：<button onclick="filter('all')">全部24条</button><button onclick="filter('dev')">dev 4条</button><button onclick="filter('train')">train 20条</button></nav>
''' + '\n'.join(cards) + '''<script>function filter(split){document.querySelectorAll('article').forEach(x=>x.hidden=split!=='all'&&x.dataset.split!==split)}</script></html>'''
    (OUT / 'review.html').write_text(document)
    print(json.dumps({k:v for k,v in assessment.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
