"""Report the cleaning trials and caption-only keep/reject pilot."""
import collections
import csv
import html
import json
import re
from p2_planning_caption_clean_pilot import ROOT, WORK, read, write, sha


def static_visual_evidence(caption):
    """Conservative pilot veto for observed missed scene/hair descriptions."""
    evidence=[]
    for sentence in re.split(r'(?<=[.!?])\s+',caption):
        if re.search(r'\b(background|backdrop)\b',sentence,re.I) and not re.search(
                r'\b(audio|hum|sound|sounds|noise|music|vocalizations)\b',sentence,re.I):
            evidence.append(dict(rule='visual_background_sentence',quote=sentence))
        if re.search(r'\bhair\b[^.!?]*\b(frames|framing|styled|pulled back)\b',sentence,re.I):
            evidence.append(dict(rule='static_hair_description',quote=sentence))
    return evidence


def main():
    rows=read(WORK/'inputs.json')
    v1=read(WORK/'review.json')
    v2=read(WORK/'v2_delete_only/results.json')
    judges=read(WORK/'v3_judge/outputs.json')
    reference=read(WORK/'v3_judge/assistant_reference.json')['rows']
    result=[]
    for r,c,d,j,ref in zip(rows,v1,v2,judges,reference):
        assert r['sample_id']==c['sample_id']==d['sample_id']==j['sample_id']==ref['sample_id']
        assert sha(r['source_path'])==r['source_sha256']
        evidence=static_visual_evidence(r['caption'])
        keep=j.get('judgment',{}).get('keep') is True and not evidence
        result.append(dict(**r,model_judgment=j.get('judgment'),model_error=j.get('error'),
            rule_evidence=evidence,filter_keep_candidate=keep,assistant_reference=ref))
    misses=[r['number'] for r in result if r['model_judgment']['keep'] and not r['assistant_reference']['expected_keep']]
    false_reject=[r['number'] for r in result if not r['model_judgment']['keep'] and r['assistant_reference']['expected_keep']]
    overrides=[r['number'] for r in result if r['model_judgment']['keep'] and not r['filter_keep_candidate']]
    kept=[r for r in result if r['filter_keep_candidate']]
    groups=[]
    for name,predicate in [('all',lambda r:True),('waveform_silent',lambda r:r['waveform_full_silence']),
            ('waveform_nonzero',lambda r:not r['waveform_full_silence']),
            ('caption_audio_positive_type',lambda r:any(t.startswith('audio:') for t in r['counted_types']))]:
        subset=[r for r in result if predicate(r)]
        groups.append(dict(group=name,total=len(subset),model_keep=sum(r['model_judgment']['keep'] for r in subset),
            final_candidate=sum(r['filter_keep_candidate'] for r in subset)))
    summary=dict(count=100,model_keep=sum(r['model_judgment']['keep'] for r in result),
        model_missed_reject_numbers=misses,model_false_reject_numbers=false_reject,
        rule_veto_numbers=overrides,candidate_count=len(kept),groups=groups,
        supplementary_detected=sum(r['contains_supplementary'] and not r['model_judgment']['keep'] for r in result),
        candidate_disagreements=[r['number'] for r in kept if not r['assistant_reference']['expected_keep']],
        source_files_unchanged=100,production_changed=False,
        caveat='Source-text review by assistant, not human/media ground truth. The rules were added after observing this same pilot; this is NOT a held-out validation or a full-data retention estimate.')
    write(WORK/'v3_judge/filter_results.json',result)
    write(WORK/'v3_judge/filter_summary.json',summary)
    # Store IDs only, never delete original samples or replace training manifests.
    write(WORK/'v3_judge/candidate_ids.json',[r['sample_id'] for r in kept])
    with (WORK/'v3_judge/filter_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['number','sample_id','split','selection_group','caption','model_keep','rule_veto','candidate_keep','defects']
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in result:
            w.writerow({**{k:r[k] for k in fields if k in r},'model_keep':r['model_judgment']['keep'],
                'rule_veto':bool(r['rule_evidence']),'candidate_keep':r['filter_keep_candidate'],
                'defects':json.dumps(r['model_judgment']['defects'],ensure_ascii=False)})
    esc=html.escape
    cards=[]
    for r,c,d in zip(result,v1,v2):
        state='keep' if r['filter_keep_candidate'] else 'reject'
        mismatch=r['model_judgment']['keep']!=r['assistant_reference']['expected_keep']
        cards.append(f'''<article data-state="{state}" data-mismatch="{str(mismatch).lower()}" id="sample-{r['number']}">
<h2>#{r['number']:03d} · {'候选保留' if state=='keep' else '候选剔除'} · {esc(r['selection_group'])}</h2><small>{esc(r['sample_id'])}</small>
<p>模型：{'保留' if r['model_judgment']['keep'] else '剔除'}；规则命中：{'是' if r['rule_evidence'] else '否'}；{('与助手逐条检查不一致' if mismatch else '与助手逐条检查一致')}</p>
<h3>原始标注（保留项仍使用此原文）</h3><p>{esc(r['caption'])}</p>
<div class="cols"><section><h3>模型给出的依据</h3><pre>{esc(json.dumps(r['model_judgment'],ensure_ascii=False,indent=2))}</pre></section>
<section><h3>规则与逐条检查</h3><pre>{esc(json.dumps(dict(rule_evidence=r['rule_evidence'],assistant_reference=r['assistant_reference']),ensure_ascii=False,indent=2))}</pre></section></div>
<details><summary>此前清洗试验：自由改写 / 原文片段删除（未采用）</summary><div class="cols"><section><h3>自由改写</h3><p>{esc(c['cleaned']['cleaned_caption'])}</p><p>自报：{esc(c['cleaned']['status'])}；同模型复核：{esc(c['audit']['verdict'])}</p></section>
<section><h3>原文片段删除</h3><p>{esc(d['cleaned_caption'] or '未输出：'+d['status'])}</p><pre>{esc(json.dumps(d['removals'],ensure_ascii=False,indent=2))}</pre></section></div></details></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>规划标签判断题 · 100条试验</title><style>body{font:16px/1.65 system-ui;max-width:1450px;padding:24px;margin:auto;background:#f4f5f7;color:#222}.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px}article{padding:22px;background:white;margin:24px 0;border:1px solid #ddd;border-radius:8px;scroll-margin-top:80px}h2{margin:0}small{overflow-wrap:anywhere;color:#555}pre{white-space:pre-wrap;font-size:14px}nav{position:sticky;top:0;background:#f4f5f7;padding:12px;z-index:1}button{padding:8px 16px;margin-right:8px}details{padding:8px 0}@media(max-width:800px){.cols{grid-template-columns:1fr}}</style>
<h1>规划标签：判断是否可原样用于SFT</h1><p>同100条原标注，Qwen3-30B-A3B-Instruct-2507只做判断，不改写。模型保留50条；助手逐条检查发现9条漏拒和2条误拒。加入针对本批漏检的背景／头发外观规则后，保留41条候选。原标签、AV缓存和训练清单均未替换。</p>
<p>本批刻意增加难例，不能据此估计全量保留比例。规则在本批结果上调整，41条不是独立测试通过率。静音、少动作本身不剔除。检查依据是原文含义，没有重新核验真实视频或音轨。</p>
<details><summary>统计及限制</summary><pre>'''+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'''</pre></details>
<nav><button onclick="f('all')">全部100条</button><button onclick="f('keep')">候选保留41条</button><button onclick="f('reject')">候选剔除59条</button><button onclick="f('mismatch')">模型判断分歧11条</button></nav>'''+''.join(cards)+'''<script>function f(v){document.querySelectorAll('article').forEach(x=>x.hidden=v==='mismatch'?x.dataset.mismatch!=='true':v!=='all'&&x.dataset.state!==v)}</script></html>'''
    (WORK/'filter_review.html').write_text(page)
    write(WORK/'v3_judge/report_protocol.json',dict(script_sha256=sha(__file__),
        judge_outputs_sha256=sha(WORK/'v3_judge/outputs.json'),reference_sha256=sha(WORK/'v3_judge/assistant_reference.json')))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
