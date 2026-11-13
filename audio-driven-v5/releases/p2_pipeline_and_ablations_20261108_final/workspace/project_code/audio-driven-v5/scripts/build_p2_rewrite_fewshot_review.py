"""Build the source-text assessment of the matched 59-caption rewrite trial."""
import collections
import csv
import html
import json
from p2_planning_caption_clean_pilot import WORK, read, write, sha

OUT=WORK/'v4_rewrite_examples'

# Qualitative checks against original captions, not model self-reported success.
BASE_GOOD={3,8,11,45,55,59,60,75,89}
BASE_BORDER={10:'将持续基本不变的表情改成中性表情，弱化口部状态和持续性；保守留待复核。'}
BASE_BAD={
    49:'仅删除Assistant前缀，仍保留眼镜、毛衣等外观。',
    53:'删了服饰，但仍保留背景墙。',
    56:'删除整个首句时丢失“全程姿态一致”的状态。',
    64:'仅改了主语，蓝色绗缝背景仍在。',
    74:'删除衣服后仍保留白墙背景。',
    83:'误删表情中性、嘴闭合、眼睛睁开及视线保持的细节。',
}
FEW_BAD={
    6:'删外观时丢失点头约在三秒发生的时间信息。',
    7:'删掉背景，但仍保留胡子、纹身、衬衫和夹克。',
    45:'误删中性表情、嘴闭合、眼部轻微变化及没有其他手臂/身体动作的描述。',
    49:'仅删除Assistant前缀，仍保留眼镜、毛衣等外观。',
    53:'错误返回unchanged，服饰和背景仍在。',
    85:'误删表情稳定、嘴眉形状无明显变化、眼睛偶有移动及手臂无动作等细节。',
    86:'错误返回unchanged，浅色背景仍在。',
}
FEW_BORDER={
    10:'删除外观时弱化“表情基本不变”的持续状态；其余主要行为、口部姿态和声音保留。',
    20:'首句从“全程大体静止、站立”压缩为“站立”，全程静止程度的限定变弱。',
    88:'完整动作顺序保留，但删除了表情变化“subtly”的强度限定；严格按保留细节规则留待复核。',
}


def evaluate(row, answer, variant, expected):
    n=row['number']
    if expected=='review':
        if answer['status']=='review':
            return 'review_routing_ok','按本轮保守规则返回复核且没有训练草稿；不代表其解释每句都正确。'
        return 'unsafe_output','原文发声状态不清，却输出可用文本；不能把缺失或不确定信息通过改写补齐。'
    if expected=='unchanged':
        if answer['status']=='review':
            return 'unnecessary_review','原文已有笑声和实际语音，“没有更多词语”不是发声矛盾；此处误拒。'
        return 'usable_original','原文可用。' + ('输出逐字保留。' if answer['status']=='unchanged' else '有非必要措辞调整，但行为及声音信息保留。')
    if variant=='no_examples':
        if n in BASE_GOOD:
            return 'rewrite_pass','删除目标外观/包装，动作、姿态、声音、数量和时序在逐条文字对照中保留。'
        if n in BASE_BORDER:
            return 'rewrite_review',BASE_BORDER[n]
        return 'rewrite_fail',BASE_BAD.get(n,'错误返回unchanged，原文外观或背景问题未解决。')
    if n in FEW_BAD:
        return 'rewrite_fail',FEW_BAD[n]
    if n in FEW_BORDER:
        return 'rewrite_review',FEW_BORDER[n]
    return 'rewrite_pass','删除目标外观/包装，动作、姿态、声音、数量和时序在逐条文字对照中保留。'


def main():
    rows=read(OUT/'inputs.json')
    outputs={(r['variant'],r['number']):r for r in read(OUT/'outputs.json')}
    reference={r['number']:r['expected_status'] for r in read(OUT/'assistant_reference.json')['rows']}
    result=[]
    for row in rows:
        assert sha(row['source_path'])==row['source_sha256']
        r=dict(number=row['number'],sample_id=row['sample_id'],split=row['split'],original=row['caption'],
            expected_status=reference[row['number']])
        for variant in ['no_examples','with_examples']:
            raw=outputs[(variant,row['number'])]
            assert raw['finish_reason']=='stop' and 'error' not in raw
            verdict,note=evaluate(row,raw['answer'],variant,r['expected_status'])
            r[variant]=dict(**raw['answer'],assessment=verdict,note=note)
        result.append(r)
    summary=dict(samples=59,expected_status=dict(collections.Counter(r['expected_status'] for r in result)),
        variants={v:dict(collections.Counter(r[v]['assessment'] for r in result)) for v in ['no_examples','with_examples']},
        source_unchanged=59,
        basis='Assistant qualitative comparison of original/generated text; same criteria for both variants. Not human blind review or verification against video/audio.',
        criteria='Preserve behaviors, posture, expression, gaze, audible/inaudible distinctions, speech, counts, intensity and timing. Remove only irrelevant visual details and chat wrappers; grammatical paraphrase allowed. Minor compression of meaningful state/intensity is review, not pass.',
        limitation='Examples targeted problems observed on this existing pilot. Results demonstrate this matched contrast, not independent generalization accuracy.')
    write(OUT/'assessment.json',dict(summary=summary,rows=result))
    candidate=[dict(sample_id=r['sample_id'],number=r['number'],caption=r['with_examples']['cleaned_caption'],
        assessment=r['with_examples']['assessment']) for r in result if r['with_examples']['assessment'] in ('rewrite_pass','usable_original')]
    write(OUT/'reviewed_candidates.json',candidate)
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['number','sample_id','expected_status','original','no_examples','with_examples','no_examples_assessment','with_examples_assessment','no_examples_note','with_examples_note']
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in result:
            w.writerow(dict(number=r['number'],sample_id=r['sample_id'],expected_status=r['expected_status'],original=r['original'],
                **{v:r[v]['cleaned_caption'] for v in ['no_examples','with_examples']},
                **{v+'_assessment':r[v]['assessment'] for v in ['no_examples','with_examples']},
                **{v+'_note':r[v]['note'] for v in ['no_examples','with_examples']}))
    esc=html.escape
    labels={'rewrite_pass':'改写候选通过','rewrite_review':'细节压缩，待复核','rewrite_fail':'改写未通过',
            'review_routing_ok':'按规则留待复核','unsafe_output':'不应输出可用草稿','usable_original':'原文可用','unnecessary_review':'误拒'}
    cards=[]
    for r in result:
        cols='<section><h3>原文</h3><p>'+esc(r['original'])+'</p></section>'
        for v,title in [('no_examples','相同规则，不带示例'),('with_examples','加入8组完整示例')]:
            a=r[v]
            cols+=f'''<section><h3>{title}</h3><b>{labels[a['assessment']]}</b><p>{esc(a['cleaned_caption']) if a['cleaned_caption'] else '模型未输出改写，返回复核。'}</p><p class="note">{esc(a['note'])}</p><details><summary>模型自报状态与理由</summary><pre>{esc(json.dumps(dict(status=a['status'],reason=a['reason']),ensure_ascii=False,indent=2))}</pre></details></section>'''
        cards.append(f'''<article id="sample-{r['number']}" data-kind="{r['expected_status']}" data-assessment="{r['with_examples']['assessment']}"><h2>#{r['number']:03d} · {esc(r['expected_status'])}</h2><small>{esc(r['sample_id'])}</small><div class="cols">{cols}</div></article>''')
    examples=read(OUT/'examples.json')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>59条问题标注 · 改写示例对照</title><style>body{font:16px/1.65 system-ui;max-width:1650px;margin:auto;padding:24px;background:#f4f5f7;color:#222}.cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}article{background:white;padding:22px;margin:24px 0;border:1px solid #ddd;border-radius:8px;scroll-margin-top:85px}small{overflow-wrap:anywhere;color:#555}h2{margin:0}pre{white-space:pre-wrap;font-size:14px}.note{background:#edf2ff;padding:12px}nav{position:sticky;top:0;background:#f4f5f7;padding:12px;z-index:1}button{padding:8px 14px;margin:3px}details{padding:8px 0}@media(max-width:950px){.cols{grid-template-columns:1fr}}</style>
<h1>59条问题标注：完整改写示例是否有用</h1><p>相同模型、system规则、输出结构、解码参数，仅示例消息不同。共118次生成，全部正常结束。按原文检查：34条适合改写，23条按本轮保守规则需复核，2条原本可用但被上一轮误拒。</p>
<p><b>34条可改写样本中，不带示例有9条通过文字对照；带示例有24条通过、3条待复核、7条未通过。</b>示例改善了结果，但仍有漏删外观和丢失行为细节；模型自报rewritten不代表合格。</p>
<p>这是助手逐条文字检查，不是人类盲评或视频真实性验证。示例针对本批已知问题设计，结果不能视为独立测试准确率。原caption、训练清单和缓存未替换。</p>
<details><summary>统计及检查口径</summary><pre>'''+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'''</pre></details>
<details><summary>统一system规则与完整示例</summary><pre>'''+esc((OUT/'system.txt').read_text()+'\n\n'+json.dumps(examples,ensure_ascii=False,indent=2))+'''</pre></details>
<nav><button onclick="f('all')">全部59条</button><button onclick="f('rewritten')">可改写34条</button><button onclick="f('pass')">示例版改写通过24条</button><button onclick="f('problems')">示例版改写问题10条</button><button onclick="f('review')">应复核23条</button><button onclick="f('unchanged')">原本可用2条</button></nav>'''+''.join(cards)+'''<script>function f(v){document.querySelectorAll('article').forEach(x=>{let yes=v==='all'||(v==='pass'?x.dataset.assessment==='rewrite_pass':v==='problems'?['rewrite_review','rewrite_fail'].includes(x.dataset.assessment):x.dataset.kind===v);x.hidden=!yes})}</script></html>'''
    (OUT/'review.html').write_text(page)
    write(OUT/'report_protocol.json',dict(script_sha256=sha(__file__),outputs_sha256=sha(OUT/'outputs.json')))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
