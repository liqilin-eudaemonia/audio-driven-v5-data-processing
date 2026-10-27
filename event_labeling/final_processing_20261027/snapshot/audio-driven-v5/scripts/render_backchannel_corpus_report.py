#!/usr/bin/env python3
"""Presentation figures and a Chinese report from the verified corpus statistics."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_listen.contracts import atomic_json,sha

LABELS={'head_nod':'点头','head_shake':'摇头','smile':'微笑','laugh':'大笑／发笑','frown':'皱眉','surprise':'惊讶','listener_vocalization':'音频回应／发声'}


def main():
    p=argparse.ArgumentParser();p.add_argument('report');args=p.parse_args();root=Path(args.report).resolve()
    s=json.loads((root/'summary.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text());assert complete['status']=='passed'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font=root/'assets/NotoSansCJKsc-Regular.otf';font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#d9e1eb','text.color':'#182b45','axes.labelcolor':'#43546c','xtick.color':'#64748b','ytick.color':'#263b54'})
    out=root/'figures';out.mkdir(exist_ok=True);N=s['scope']['samples'];classes=s['classes']['open'];cats=list(LABELS);t=s['totals'];total_hours=s['scope']['hours'];empty=N-t['open_with_events']
    blue='#326cc0';orange='#db8d2d';gray='#a8b3c2';green='#3d948a'
    def save(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight')
        plt.close(fig)
    def clean(ax):
        ax.set_axisbelow(True);ax.grid(axis='x',color='#e7ecf3',linewidth=.8);ax.tick_params(axis='both',length=0,pad=9)
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.05,.95,'全量倾听数据：Backchannel 标注概览',fontsize=25,weight='bold')
    fig.text(.05,.91,'后处理后的开放式七类事件 · 原始 listener 预裁框 ≥384×384 · 所有片段保留',fontsize=12,color='#64748b')
    metrics=[('片段总数',f'{N:,}'),('数据总时长',f'{total_hours:,.2f} h'),('有七类事件记录',f'{t["open_with_events"]:,}'),('未检出七类事件',f'{empty:,}')]
    for i,(label,value) in enumerate(metrics):
        x=.05+i*.235;fig.text(x,.84,label,fontsize=12,color='#64748b');fig.text(x,.785,value,fontsize=27,weight='bold',color=blue if i<3 else '#67758b')
    ax=fig.add_axes([.115,.18,.40,.49]);vals=[classes[c]['prevalence_percent'] for c in cats];bars=ax.barh(np.arange(7),vals,color=[blue]*6+[orange],height=.57)
    ax.set_yticks(np.arange(7),[LABELS[c] for c in cats]);ax.invert_yaxis();ax.set_xlim(0,66);ax.set_xlabel('占全部片段的比例（%）');ax.set_title('七类覆盖：同一片段可包含多类',loc='left',fontsize=16,pad=20);clean(ax)
    for i,c in enumerate(cats):ax.text(vals[i]+.8,i,f'{classes[c]["samples"]:,}  ·  {vals[i]:.2f}%',va='center',fontsize=11)
    ax=fig.add_axes([.69,.22,.26,.43]);keys=['visual_only','audio_only','both','neither'];names=['仅视觉事件','仅音频事件','视听事件均有','无七类事件'];v=[s['modality']['open'].get(k,0) for k in keys]
    ax.barh(range(4),[100*x/N for x in v],color=[blue,orange,green,gray],height=.55);ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlim(0,70);ax.set_xlabel('占全部片段的比例（%）');ax.set_title('片段内的视听事件分布',loc='left',fontsize=16,pad=20);clean(ax)
    for i,x in enumerate(v):ax.text(100*x/N+.8,i,f'{x:,}\n{100*x/N:.2f}%',va='center',fontsize=10.5)
    fig.text(.05,.075,'统计口径：模型候选记录，不是人工确认的真实发生次数；未检出事件不等于确认没有回应。',fontsize=11,color='#64748b')
    fig.text(.05,.04,'“视听事件均有”表示同一片段内存在两种模态的事件，不要求时间重叠。统计日期：2026-10-27。',fontsize=10,color='#64748b')
    save(fig,'01_overview')
    fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor='white');fig.subplots_adjust(left=.08,right=.97,top=.78,bottom=.20,wspace=.35)
    fig.suptitle('事件丰富度与未检出事件片段',x=.05,ha='left',fontsize=24,weight='bold',y=.97)
    fig.text(.05,.88,f'事件条目共 {t["open_events"]:,} 条；同类持续状态可能有多条描述，条目数不能直接解释为动作次数。',fontsize=12,color='#64748b')
    keys=['0','1','2–5','6–10','11–20','>20'];v=[s['event_count_histogram'].get(k,0) for k in keys];ax=axes[0]
    ax.bar(range(6),v,color=[gray]+[blue]*5,width=.65);ax.set_xticks(range(6),keys);ax.set_xlabel('每段七类事件条目数');ax.set_ylabel('片段数');ax.spines['bottom'].set_visible(False);ax.grid(axis='y',color='#e7ecf3');ax.set_axisbelow(True);ax.set_ylim(0,max(v)*1.25)
    for i,x in enumerate(v):ax.text(i,x+max(v)*.025,f'{x:,}\n{100*x/N:.1f}%',ha='center',fontsize=10)
    q=s['buckets'].get('quiet_candidate',{}).get('samples',0);u=empty-q;ax=axes[1]
    ax.barh([0,1],[q,u],color=[green,gray],height=.5);ax.set_yticks([0,1],['安静倾听候选','证据仍有不完整／不确定']);ax.invert_yaxis();ax.set_xlim(0,max(q,u)*1.5);clean(ax);ax.set_xlabel('未检出七类事件的片段数');ax.set_title(f'无事件记录的 {empty:,} 段：全部保留',loc='left',fontsize=15,pad=20)
    for i,x in enumerate([q,u]):ax.text(x+max(q,u)*.025,i,f'{x:,}\n占无事件片段 {100*x/empty:.1f}%',va='center',fontsize=11)
    fig.text(.05,.08,'“安静倾听候选”：视听整段回执有效、无记录的推理或语义错误、无待复核／未知标记，且未检出七类事件。',fontsize=10.5,color='#64748b')
    fig.text(.05,.035,'注视、单次转头等 out_of_scope 描述不作为失败；两组都不是人工确认的“无回应”真值，也没有自动改成负标签。',fontsize=10.5,color='#64748b')
    save(fig,'02_event_distribution_and_no_event')
    fig,ax=plt.subplots(figsize=(11,9),facecolor='white');fig.subplots_adjust(left=.20,bottom=.22,right=.92,top=.85)
    values=np.asarray(s['cooccurrence'],dtype=float)/N*100
    im=ax.imshow(values,cmap='Blues',vmin=0,vmax=max(vals));ax.set_xticks(range(7),[LABELS[c] for c in cats],rotation=35,ha='right');ax.set_yticks(range(7),[LABELS[c] for c in cats]);ax.tick_params(length=0)
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center',fontsize=11,color='white' if values[i,j]>24 else '#182b45')
    fig.colorbar(im,ax=ax,shrink=.72,label='占全量片段的比例（%）');fig.suptitle('七类事件在同一片段内的共现',fontsize=22,weight='bold',y=.97)
    fig.text(.07,.055,'对角线为各类覆盖率；非对角线表示两类都在该片段出现，不代表二者同时发生。',fontsize=11,color='#64748b');save(fig,'03_class_cooccurrence')
    # Report text is generated from the same verified summary as the figures.
    rows=['| 类别 | 含该类的片段数 | 全量占比 | 候选事件条目数 |','|---|---:|---:|---:|']
    for c in cats:
        x=classes[c];rows.append(f'| {LABELS[c]} | {x["samples"]:,} | {x["prevalence_percent"]:.2f}% | {x["events"]:,} |')
    canonical=['| 类别 | 开放式记录：片段数 | canonical：片段数 | canonical：事件数 |','|---|---:|---:|---:|']
    for c in cats:
        x=s['classes']['canonical'][c];canonical.append(f'| {LABELS[c]} | {classes[c]["samples"]:,} | {x["samples"]:,} | {x["events"]:,} |')
    combos=['| 类别组合（恰好这些类别） | 片段数 | 全量占比 |','|---|---:|---:|']
    for x in s['combinations']['open'][:10]:combos.append(f'| {"＋".join(LABELS[c] for c in x["classes"]) or "未检出七类事件"} | {x["samples"]:,} | {x["percent"]:.2f}% |')
    modes=['| 模态情况 | 片段数 | 全量占比 |','|---|---:|---:|']
    for key,name in zip(['visual_only','audio_only','both','neither'],['仅视觉事件','仅音频事件','同片段视听事件均有','无七类事件记录']):
        n=s['modality']['open'].get(key,0);modes.append(f'| {name} | {n:,} | {100*n/N:.2f}% |')
    corpus=['| 来源 | 片段数 | 时长 | 有事件的片段数 |','|---|---:|---:|---:|']
    for name,x in s['by_corpus'].items():corpus.append(f'| {name} | {x["samples"]:,} | {x["seconds"]/3600:,.2f} h | {x["with_open_events"]:,} |')
    quiet=s['buckets']['quiet_candidate'];uncertain=s['buckets']['no_events_insufficient_evidence'];a=s['audio_postprocess_audit']
    import sqlite3
    with sqlite3.connect(f'file:{root}/sample_index.sqlite?mode=ro',uri=True) as db:
        unadmitted=dict(db.execute('SELECT open_events>0,count(*) FROM samples WHERE training_admitted=0 GROUP BY open_events>0'))
    text=f'''# 全量后处理数据 Backchannel 统计（2026-10-27）

## 可直接用于汇报的摘要

本轮在原始 listener 预裁框不小于 384×384 的数据上，完成 **{N:,} 段、{total_hours:,.2f} 小时**倾听数据的多模型标注、信息整合及音频后处理。后处理后的开放式七类记录共包含 **{t['open_events']:,} 条候选事件**，覆盖 **{t['open_with_events']:,} 段（{100*t['open_with_events']/N:.2f}%）**；另外 **{empty:,} 段（{100*empty/N:.2f}%）未检出七类事件，全部保留**。数据覆盖六类视觉反应及一类音频回应，微笑最常见，摇头最稀有。所有结果均为模型弱标注，不作为人工真值或模型准确率结论。

![全量概览](figures/01_overview.png)

## 一、范围与口径

- 只统计全量后处理目录中的 {N:,} 个唯一 train ID；不混入 50h 子集、独立 test 24 条或其他重复小样本。
- 这是用户后续确认的 crop≥384 范围，不是最初全部 1,419,124 段。时长是片段时长相加，不是源会话时间去重后的小时数。
- 平均片段长度 {s['scope']['seconds']/N:.3f} 秒，中位数 {s['quantiles']['seconds']['50']:.3f} 秒。
- 本报告正文采用**后处理后的开放式七类事件记录**。canonical 是经过训练监督规则筛选的另一层标签，单列对照，避免混用。
- “事件数”是整合后的模型候选条目数；持续微笑等可被描述为多条，不能宣称真实发生了同样多次动作。未重新修改合并规则或描述原文。
- 音频只计算过滤后保留的区间；同一事件被静音切成多个支持区间时只计一次。重叠区间在时长统计中取并集。
- 没有删除任何片段，也没有修改训练准入或把未知改成负标签。全量 ID 在 [sample_ids_all.txt](sample_ids_all.txt)，无事件 ID 在 [no_open_event_ids.txt](no_open_event_ids.txt)。

{chr(10).join(corpus)}

## 二、七类覆盖

{chr(10).join(rows)}

一段可以含多类，所以各类片段数不能相加当作总片段数。大笑／发笑属于视觉类；音轨中的笑声计入音频回应／发声。

## 三、模态与丰富度

{chr(10).join(modes)}

上述视听共现只要求同片段两种模态都有事件。进一步按实际时间区间求交，有 {t['open_with_temporal_av_overlap']:,} 段存在视听事件时间重叠，累计重叠支持时长 {t['open_temporal_av_overlap_seconds']/3600:.2f} 小时。

每段候选事件数：均值 {t['open_events']/N:.2f}，中位数 {s['quantiles']['open_events']['50']:.0f}，p90={s['quantiles']['open_events']['90']:.0f}，p95={s['quantiles']['open_events']['95']:.0f}。条目多不等同于回应质量高，不能单凭该值评价模型效果。

![事件数与无事件片段](figures/02_event_distribution_and_no_event.png)

最常见的互斥类别组合：

{chr(10).join(combos)}

![七类共现](figures/03_class_cooccurrence.png)

## 四、无事件片段全部保留

共有 **{empty:,} 段、{quiet['hours']+uncertain['hours']:.2f} 小时**未检出七类事件。其中：

- **{quiet['samples']:,} 段、{quiet['hours']:.2f} 小时**：视听整段回执均有效，没有记录的推理／语义错误、待复核观察或未知区间，可作为“安静倾听候选”。仍然不是人工确认的无回应真值。
- **{uncertain['samples']:,} 段、{uncertain['hours']:.2f} 小时**：存在回执、待复核或未知状态等证据限制，继续保留，不能直接当作确认负例。

`unresolved_observations` 是上游容器名，其中 `out_of_scope` 只表示注视、单次转头等描述不属于当前七类，**本报告没有把它当成标注失败**。真正的 `needs_review` 或其他未决原因则保留为不确定性。

960 段未获原训练标签准入的样本也全部保留在统计中，见 [unadmitted_ids.txt](unadmitted_ids.txt)。其中 **{unadmitted.get(1,0):,} 段仍有开放式事件，{unadmitted.get(0,0):,} 段没有七类事件记录**，与“没有回应”的集合不是同一概念。当前训练读取仍遵守原标记，本次没有更改训练损失或数据读取准入逻辑。

## 五、音频后处理与 canonical 对照

- {a['changed_audio']:,} 段更新音频，{a['unchanged_audio']:,} 段原样复用；没有生成新的静音版 WAV。
- {a['full_silence_support']:,} 段按官方区间规则得到全静音支持。这个数与“没有音频事件标签”不同：有保留的官方发声区间，也可能没有模型事件标签。
- 开放式保留音频事件 {a['open_retained_audio_events']:,} 条，覆盖 {classes['listener_vocalization']['samples']:,} 段；canonical 保留 {a['retained_audio_events']:,} 个音频源事件，覆盖 {a['with_audio_labels']:,} 段。
- canonical 共 {t['canonical_events']:,} 个源事件；序列化时间段条目为 {t['canonical_interval_entries']:,} 条。两者差 {t['canonical_interval_entries']-t['canonical_events']:,} 条来自同一音频事件的分段支持，不计为新增发声。

{chr(10).join(canonical)}

## 六、汇报文件与复现

- [概览 PNG](figures/01_overview.png)／[SVG](figures/01_overview.svg)／[PDF](figures/01_overview.pdf)：可直接插入幻灯片。
- [分布 PNG](figures/02_event_distribution_and_no_event.png)／[SVG](figures/02_event_distribution_and_no_event.svg)；[共现 PNG](figures/03_class_cooccurrence.png)／[SVG](figures/03_class_cooccurrence.svg)。
- [七类汇总 CSV](class_statistics.csv)、[逐片段 CSV](samples.csv)、[完整统计 JSON](summary.json)、[完成与校验](COMPLETE.json)。
- [统计代码](../../scripts/report_postprocessed_backchannels.py)、[图表与本文生成代码](../../scripts/render_backchannel_corpus_report.py)。

读取并核验 20,666 个源任务，逐条校验开放记录和 canonical 身份，最终 ID 与 scope 完全一致；canonical 七类覆盖、音频数量和准入数量与原全量验收精确对齐。报告没有改写上游数据。源验收 SHA：`{s['source_audit_sha256']}`。

```bash
python scripts/report_postprocessed_backchannels.py \\
  --run experiments/p2_full_audio_postprocess_crop384_20261026 \\
  --output experiments/p2_full_backchannel_report_20261027 --workers 16
python scripts/render_backchannel_corpus_report.py experiments/p2_full_backchannel_report_20261027
```
'''
    (root/'README.md').write_text(text)
    (root/'presentation_summary.txt').write_text(text.split('## 可直接用于汇报的摘要\n\n',1)[1].split('\n\n![',1)[0]+'\n')
    with (root/'七类覆盖汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['统计口径','类别','含该类片段数','占全量比例（%）','候选源事件条目数','区间并集时长（秒）'])
        for scope,name in [('open','后处理开放式记录'),('canonical','canonical训练标签')]:
            for c in cats:
                x=s['classes'][scope][c];w.writerow([name,LABELS[c],x['samples'],round(x['prevalence_percent'],4),x['events'],round(x['union_seconds'],3)])
    atomic_json(root/'presentation_files.json',dict(source_summary_sha256=sha(root/'summary.json'),files={str(f.relative_to(root)):sha(f) for f in [root/'README.md',root/'presentation_summary.txt',root/'七类覆盖汇总.csv',*sorted(out.iterdir())]}))
    print(json.dumps(dict(report=str(root/'README.md'),figures=3,samples=N),ensure_ascii=False))


if __name__=='__main__':main()
