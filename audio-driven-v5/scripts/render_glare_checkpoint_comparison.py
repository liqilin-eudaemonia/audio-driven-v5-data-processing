"""Render completed raw metricJSON into the user's single-table comparisons."""
import datetime
import csv
import html
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'experiments'
DOC=ROOT/'evaluation/VICO_ALL_LORA6000_6500_RESULTS_20261112.md'
FINAL=EX/'glare_final_vico43_responsenet586_20261112'
EARLY=EX/'glare19050_vico43_responsenet586_seamless200_20261112'
FULL=EX/'glare_final_seamless_test6624_20261112'
EARLYFULL=EX/'glare19050_seamless_test6624_20261112'
LABELS=['LoRA6000＋SFT2752','LoRA6500＋SFT2752','GLARE final32940','GLARE step19050']

def read(path):return json.loads(path.read_text()) if path.exists() else None

def cells(values,scores,formatting=None):
    valid=[s for v,s in zip(values,scores) if v is not None and math.isfinite(s)]
    levels=sorted(set(valid)) if len(valid)>=2 else []
    out=[]
    for i,(value,score) in enumerate(zip(values,scores)):
        if value is None:out.append('—');continue
        text=formatting(i,value) if formatting else f'{value:.6f}'
        if levels and score==levels[0]:text=f'**{text}**'
        elif len(levels)>1 and score==levels[1]:text=f'<u>{text}</u>'
        out.append(text)
    return out

def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
                     ['| '+' | '.join(row)+' |' for row in rows])

def render_html():
    """Export only the document's tables, retaining ranks and section names."""
    from render_test256_checkpoint_tables import ANCHORS, NOTES, sources
    source=DOC.read_text()
    sections=[];csv_rows=[]
    title='测评结果'
    lines=source.splitlines()
    def inline(value):
        value=html.escape(value,quote=False)
        value=re.sub(r'&lt;br\s*/?&gt;','<br>',value)
        value=re.sub(r'&lt;u&gt;(.*?)&lt;/u&gt;',r'<u>\1</u>',value)
        return re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',value)
    for index,line in enumerate(lines):
        if line.startswith('## '):title=line[3:]
        if not line.startswith('|') or (index and lines[index-1].startswith('|')):continue
        table=[]
        for row in lines[index:]:
            if not row.startswith('|'):break
            table.append([cell.strip() for cell in row.strip('|').split('|')])
        assert len(table)>=2 and all(re.fullmatch(r':?-{3,}:?',v) for v in table[1])
        assert all(len(row)==len(table[0]) for row in table)
        headings=''.join(f'<th scope="col">{inline(cell)}</th>' for cell in table[0])
        for row in table[2:]:
            for metric,value in zip(table[0][1:],row[1:]):
                plain=re.sub(r'<br\s*/?>',' ',value).replace('**','').replace('<u>','').replace('</u>','')
                csv_rows.append(dict(scope=title,method=row[0],metric=re.sub(r'<br\s*/?>',' ',metric),display_value=plain,rank=1 if '**' in value else 2 if '<u>' in value else ''))
        body=''.join('<tr>'+''.join(
            f'<th scope="row">{inline(cell)}</th>' if col==0 else f'<td>{inline(cell)}</td>'
            for col,cell in enumerate(row))+'</tr>' for row in table[2:])
        ident=ANCHORS.get(title,f'table-{len(sections)+1}')
        note=f'<p class="scope-note">{html.escape(NOTES[title])}</p>' if title in NOTES else ''
        sections.append(f'<section><h2 id="{ident}">{html.escape(title)}</h2>'
                        +note+
                        f'<div class="table-scroll" tabindex="0" role="region" aria-labelledby="{ident}">'
                        f'<table aria-labelledby="{ident}"><thead><tr>{headings}</tr></thead>'
                        f'<tbody>{body}</tbody></table></div></section>')
    assert sections,'No tables in the source document'
    style='''
*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:#18202b;
font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{padding:24px;max-width:1920px;margin:auto}section{margin-bottom:28px}
h2{font-size:17px;font-weight:650;margin:0 0 12px}
.scope-note{font-size:14px;line-height:1.7;max-width:1250px;color:#435065}nav{margin:16px 0 24px;display:flex;gap:16px;flex-wrap:wrap}a{color:#1558a6}h2{scroll-margin-top:16px}
.table-scroll{overflow:auto;max-height:78vh;border:1px solid #dce1e8;
border-radius:8px;background:white;scrollbar-gutter:stable;outline-offset:3px}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;
font-size:13px;font-variant-numeric:tabular-nums;line-height:1.5}
th,td{padding:11px 14px;border-bottom:1px solid #e7eaf0;white-space:nowrap;text-align:right}
thead th{position:sticky;top:0;z-index:2;background:#eef1f5;font-weight:600}
tr>:first-child{position:sticky;left:0;z-index:1;text-align:left;min-width:250px;
max-width:290px;white-space:normal;border-right:1px solid #dce1e8}
tbody th{background:white;font-weight:500}thead th:first-child{z-index:3}
tbody tr:nth-child(even)>*{background:#fafbfd}tbody tr:hover>*{background:#f0f5fb}
tbody tr:last-child>*{border-bottom:0}strong{font-weight:800}
u{text-underline-offset:3px;text-decoration-thickness:1.5px}
@media(max-width:600px){main{padding:14px 10px}h2{font-size:16px}
th,td{padding:10px}tr>:first-child{min-width:170px;max-width:170px}}
@media print{body{background:white}main{padding:0}.table-scroll{max-height:none;overflow:visible}
table{font-size:8px}th,td{padding:4px}th,tr>:first-child{position:static}}
'''
    page='<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'+\
         '<meta name="viewport" content="width=device-width,initial-scale=1">'+\
         '<title>测评结果表格</title><style>'+style+'</style></head><body><main>'+\
         '<p><strong>第一名加粗</strong>；<u>第二名下划线</u>。各测试范围分别排名；未完成不填0、不排名。GLARE没有listener音频，音频指标N/A。</p>'+\
         '<nav>'+''.join(f'<a href="#{ident}">{html.escape(title)}</a>' for title,ident in ANCHORS.items())+'</nav>'+\
         '\n'.join(sections)+'</main></body></html>\n'
    output=DOC.with_suffix('.html')
    temp=output.with_suffix('.html.tmp');temp.write_text(page);temp.replace(output)
    public=ROOT/'data_previews/listen_benchmark_review/p2_responsenet_random200_20261105'
    if public.exists():
        target=public/'checkpoint_results.html';temp=target.with_suffix('.tmp');temp.write_text(page);temp.replace(target)
        target=public/'checkpoint_results_display.csv';temp=target.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['scope','method','metric','display_value','rank']);writer.writeheader();writer.writerows(csv_rows)
        temp.replace(target)
        raw={}
        for folder in [FINAL,EARLY,FULL,EARLYFULL]:
            for name in ['responsenet/RESULTS.json','seamless/RESULTS.json','vico/IMAGE_RESULTS.json','vico/DIM_RESULTS.json','vico/bfm/RESULTS.json','fixed200_RESULTS.json','test6624_RESULTS.json','status.json','COMPLETE.json']:
                path=folder/name
                if path.exists():raw[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data=read(path))
        for path in sources():
            raw[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data=read(path))
        target=public/'checkpoint_results_raw.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n');temp.replace(target)
    return output

def vico():
    models=[]
    labels=LABELS.copy()
    for step in [6000,6500]:
        j=read(EX/f'p2_vico43_all_lora{step}_sft2752_20261111/ALL_RESULTS.json')
        models.append((j['official_image_identity5'],j['DIM_EMOCA12'],j['CustomListener_public_definitions']))
    for folder in [FINAL,EARLY]:models.append(tuple(read(folder/f'vico/{f}') for f in ['IMAGE_RESULTS.json','DIM_RESULTS.json','bfm/RESULTS.json']))
    # These rows are part of the shared comparison, including on later GLARE
    # refreshes. Their EMOCA results must not be dropped by a four-model refresh.
    reference=EX/'p2_vico_avatarforcing_added_20261111'
    motion=read(reference/'MOTION_RESULTS.json')['methods']
    with (reference/'image_metrics.csv').open() as stream:
        image={row['method']:row for row in csv.DictReader(stream)}
    names=[('joint','joint（历史联合模型）'),
           ('joint_sft2752','joint_sft2752（历史联合模型）'),
           ('listenformer','ListenFormer (local retraining)'),
           ('dystream','DyStream'),
           *[(f'l2l_gpavatar_{name}',f'L2L {name} + GPAvatar')
             for name in ['conan','fallon','stephen','trevor']],
           ('avatarforcing_audio_only','Avatar Forcing (audio-only adaptation)')]
    for key,label in names:
        result=motion[key]
        raw_image=image.get(label)
        im=({'metrics':{key:float(raw_image[key])
                       for key in ['psnr','ssim','fid','csim','cpbd']}}
            if raw_image else None)
        dim={'frames':result['frames'],'models':{'ours':{'primary':result['metrics']}}}
        models.append((im,dim,None))
        labels.append(label)
    defs=[]
    for k,direction in [('psnr','up'),('ssim','up'),('fid','down'),('csim','up'),('cpbd','up')]:
        defs.append((lambda m,k=k:m[0]['metrics'][k] if m[0] else None,direction,None))
    for k,direction in [('FD','down'),('P_FD','down'),('MSE','down'),('SID','gt'),('Var','gt'),('rPCC','up')]:
        for g in ['exp','pose']:
            defs.append((lambda m,k=k,g=g:m[1]['models']['ours']['primary'][k][g] if m[1] else None,direction,
                         (lambda m,k=k,g=g:m[1]['models']['ours']['primary'][k+'_GT'][g]) if direction=='gt' else None))
    for k,groups,direction in [('FD',['exp','angle','trans'],'down'),('RTLCC',['exp','pose'],'down'),('RWTLCC',['exp','pose'],'down'),
                                ('FID_delta_fm',['exp','pose'],'down'),('SND',['exp','pose'],'down'),('V_D',['exp','pose'],'gt')]:
        for g in groups:
            defs.append((lambda m,k=k,g=g:m[2]['metrics_table_scale'][k][g] if m[2] else None,direction,
                         (lambda m,g=g:m[2]['metrics_table_scale']['V_D_GT'][g]) if direction=='gt' else None))
    assert len(defs)==30
    columns=[]
    for getter,direction,gt in defs:
        values=[getter(m) for m in models]
        scores=[math.inf if v is None else -v if direction=='up' else abs(v-gt(m)) if direction=='gt' else v for v,m in zip(values,models)]
        columns.append(cells(values,scores))
    return [[labels[i],f"{m[1]['frames']:,}" if m[1] else '待评分',*[c[i] for c in columns]] for i,m in enumerate(models)]

VIDEO=[('psnr','up'),('ssim','up'),('lpips','down'),('fid','down'),('fvd','down'),('emo_acc','up'),('emo_div_generated','gt'),('delta_resp','down')]

def video(results,states):
    columns=[]
    for key,direction in VIDEO:
        values=[r['metrics'].get(key) if r else None for r in results]
        scores=[math.inf if v is None else -v if direction=='up' else abs(v-r['metrics']['emo_div_gt']) if direction=='gt' else v for v,r in zip(values,results)]
        def formatting(i,value):
            if key=='emo_acc':return f'{value*100:.4f}%'
            if direction=='gt':return f"{value:.6f}／{results[i]['metrics']['emo_div_gt']:.6f}"
            return f'{value:.6f}'
        columns.append(cells(values,scores,formatting))
    out=[]
    for i,r in enumerate(results):
        coverage='—' if not r else f"{r['coverage']['psnr']['pairs']}／{r['coverage']['emotion']['video_identities']}／{r['coverage']['delta_resp']['valid_phases']}"
        out.append([LABELS[i],states[i],*[c[i] for c in columns],coverage])
    return out

def render():
    text=DOC.read_text();tables={};rwork=EX/'p2_responsenet_full586_all_lora6000_6500_20261111';state=read(rwork/'status.json')
    results=[];states=[]
    for step in [6000,6500]:
        r=read(rwork/f'step_{step:06d}/RESULTS.json')
        if r:r=r.get(f'all_lora{step}_sft2752/responsenet',r)
        results.append(r);states.append('已完成' if r else f"生成中（视频{state.get('videos',0)}/586）" if state.get('step')==step else '排队／未完成')
    for folder in [FINAL,EARLY]:
        r=read(folder/'responsenet/RESULTS.json');results.append(r);states.append('已完成' if r else '评分中')
    tables['ResponseNet：完整586条']=video(results,states)
    lora=read(EX/'p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/RESULTS.json')['ltx_all_lora_event_feature_whisper_6000_sft2752/seamless']
    early=read(EARLY/'seamless/RESULTS.json')
    tables['Seamless：固定诊断200条']=video([lora,None,read(FULL/'fixed200_RESULTS.json'),early],
        ['已完成（支持不足见覆盖）','尚无同范围成绩','已完成（全量任务已暂停）' if read(FULL/'status.json').get('stage')=='paused' else '已完成（全量任务继续）','已完成' if early else '评分中'])
    full=read(FULL/'test6624_RESULTS.json') if (FULL/'COMPLETE.json').exists() else None;live=read(FULL/'status.json')
    efull=read(EARLYFULL/'test6624_RESULTS.json') if (EARLYFULL/'COMPLETE.json').exists() else None;elive=read(EARLYFULL/'status.json') or {}
    tables['Seamless：完整test6624条']=video([None,None,full,efull],['尚无同范围成绩','尚无同范围成绩',
        '已完成（覆盖见结果）' if full else f"{'已暂停' if live.get('stage')=='paused' else '处理中'}（生成{live.get('generated',0)}/6624；已完整评分{live.get('scored_in_complete_blocks',0)}条）",
        '已完成（覆盖见结果）' if efull else f"处理中（生成{elive.get('generated',200)}/6624；已完整评分{elive.get('scored_in_complete_blocks',200)}条）"])
    tables['ViCo：正式test43']=vico()
    for section,rows in tables.items():
        pattern=r'(## '+re.escape(section)+r'\n)(.*?)(?=\n## |\Z)';match=re.search(pattern,text,re.S);assert match,section
        body=match.group(2);block=re.search(r'^\| 方法／权重.*(?:\n\|.*)+',body,re.M);assert block,section
        headers=[c.strip() for c in block.group(0).splitlines()[0].strip('|').split('|')]
        assert all(len(row)==len(headers) for row in rows),section
        body=body[:block.start()]+markdown(headers,rows)+body[block.end():]
        text=text[:match.start(2)]+body+text[match.end(2):]
    text=text.replace('LoRA6000／LoRA6500／GLARE32940\n','LoRA6000／LoRA6500／GLARE32940／GLARE19050\n')
    text=re.sub(r'更新：[^。]+。','更新：'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')+'。',text,count=1)
    text=text.replace('三组均43/43条','表内四组均43/43条').replace('GLARE与6000的逐ID','GLARE final32940与6000的逐ID')
    text=text.replace('没有重算三模型交集，三组整体不是严格同帧对照','没有重算四模型交集，四组整体不是严格同帧对照')
    text=text.replace('三组均512／25fps生成','各组均512／25fps生成')
    text=text.replace('当前仅GLARE有全量成绩，因此不标第一／第二名。','仅对已有完整成绩的方法标名次，未完成方法不参与。')
    if 'GLARE step19050使用' not in text:
        text=text.replace('## ResponseNet：完整586条','GLARE step19050使用用户指定的`before_extension30epochs.pt`，829条输入／韵律缓存全部复用。共享GPU时LIA解码batch4，final32940为batch8，FP32批处理可能有通常的微小舍入差异。新权重的DIM有效帧和原始JSON见[运行与结果](../experiments/glare19050_vico43_responsenet586_seamless200_20261112/README.md)。\n\n## ResponseNet：完整586条')
    from render_test256_checkpoint_tables import TITLES, build_sections
    for title in TITLES.values():
        text=re.sub(r'## '+re.escape(title)+r'\n.*?(?=\n## |\Z)','',text,flags=re.S)
    text=text.replace('## ViCo：正式test43',build_sections(cells,markdown)+'## ViCo：正式test43',1)
    temp=DOC.with_suffix('.tmp');temp.write_text(text);temp.replace(DOC)
    render_html()
    return DOC

if __name__=='__main__':print(render())
