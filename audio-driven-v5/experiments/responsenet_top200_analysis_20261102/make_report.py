"""Read-only, locally browsable score report with lazy media players."""
import html,json,os
from pathlib import Path
from urllib.parse import quote
BASE=Path(__file__).resolve().parent
s=json.loads((BASE/'summary.json').read_text());rs=[json.loads(l) for l in (BASE/'top200.jsonl').read_text().splitlines()]
a,t=s['all586'],s['top200'];e=html.escape
ARMS=s['method']['arms'];names={'av_event_6000':'AV＋Event 6000','av_event_8927':'AV＋Event 8927','av_only_6000':'仅AV 6000','av_only_8927':'仅AV 8927'}
def link(path):
 p=Path(path);assert p.is_file(),p
 return e(quote(os.path.relpath(p,BASE),safe='/'),quote=True)
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+e(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ResponseNet 四权重平均视频综合分 Top200</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#182330;background:#f5f7fa;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:22px;margin-top:30px}.box,details{background:white;border:1px solid #dce3eb;border-radius:8px;padding:16px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #e3e7ec}th{background:#edf2f7}a{color:#155dad}summary{cursor:pointer;font-weight:600}.players{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:10px;overflow:auto}video,audio{width:100%;max-width:100%}video{background:#151515;aspect-ratio:1}.note{color:#556274;font-size:14px}.badge{padding:3px 7px;background:#e8f3ed;color:#286548;border-radius:4px}code{overflow-wrap:anywhere} @media(max-width:900px){.players{grid-template-columns:repeat(2,minmax(140px,1fr))}}
</style><main><h1>ResponseNet：四权重平均视频综合分 Top200</h1>''']
parts.append('<p>基于已完成的586个pair × 4个权重，仅分析已有结果。<a href="top200.csv" download>下载200条CSV</a> · <a href="top200.jsonl">完整逐条数据</a> · <a href="summary.json">统计与方法</a></p>')
parts.append(f'<div class="box">200条来自 <b>{t["source_count"]}</b> 个源样本，共 <b>{t["total_minutes"]:.2f}分钟</b>，平均 <b>{t["duration_mean_s"]:.2f}秒</b>。其中 <b>{t["gt_zero"]}条GT音轨全零</b>，{t["gt_nonzero"]}条非零；自动检出声音活动{t["gt_acoustic"]}条、语音{t["gt_speech"]}条。</div>')
parts.append('<h2>排序口径</h2><p>先对每个pair的四权重指标取算术均值，再对PSNR↑、SSIM↑、LPIPS↓、表情准确率↑、ΔResp↓统一方向，转成586条内的百分位，五项各占20%。分数为0–100的相对视频综合分，不是准确率，也不是官方现成综合指标。</p><p class="note">FID/FVD是集合指标，未纳入逐条排序；Emo-Div的生成类别数并非越大越好，未纳入；UTMOS全部N/A。排名不包含音频质量。原586条清单、推理队列和正式评测结果未修改。</p>')
parts.append('<h2>指标概览</h2>')
labels={'psnr':'PSNR ↑','ssim':'SSIM ↑','lpips':'LPIPS ↓','emo_acc':'表情准确率 ↑','delta_resp':'ΔResp ↓'}
parts.append(table(['指标','全586条','Top200'],[(labels[m],f'{a["metric_original_aggregation_mean4"][m]:.4f}',f'{t["metric_original_aggregation_mean4"][m]:.4f}') for m in labels]))
parts.append('<p class="note">此表复用原评测的帧数／时长加权汇总，再平均四权重，与原RESULTS.json的全量结果逐项核对一致；排名本身是逐pair计算。</p>')
parts.append('<h2>这200条的特点</h2>')
parts.append(table(['特征','全586条','Top200'],[
 ('来源数',a['source_count'],t['source_count']),('平均时长',f'{a["duration_mean_s"]:.2f}s',f'{t["duration_mean_s"]:.2f}s'),
 ('GT全零音轨',f'{a["gt_zero"]} / 586',f'{t["gt_zero"]} / 200'),('GT非零音轨',a['gt_nonzero'],t['gt_nonzero']),
 ('自动识别为中性的主要表情',f'{a["dominant_emotion_proxy"].get("neutral",0)} / 586',f'{t["dominant_emotion_proxy"].get("neutral",0)} / 200'),
 ('GT关键点运动中位数',f'{a["gt_motion_median"]:.4f}',f'{t["gt_motion_median"]:.4f}'),
 ('全量运动量最高四分位的样本',a['gt_motion_quartiles_global'][3],t['gt_motion_quartiles_global'][3]),
 ('官方词时间来源',a['asr_groups'].get('official_word_time',0),t['asr_groups'].get('official_word_time',0)),
 ('本地补充词时间来源',a['asr_groups'].get('local_word_time',0),t['asr_groups'].get('local_word_time',0))]))
parts.append('<p>主要偏向中性表情、较小运动量，时长没有明显变短。表情来自DeepFace自动分类；运动量是共同有效帧上眉眼关键点的运动代理，均未做逐条人工确认。</p>')
parts.append(f'<p>四个权重各自Top200与本名单的交集为 {list(s["sensitivity"]["individual_top200_overlaps"].values())}；其中 <b>{s["sensitivity"]["top200_in_all_four_individual_top200"]}条</b>在四个权重中均进入Top200。若改成“画面相似度／表情／动作”三类等权，重合{s["sensitivity"]["balanced_visual_emotion_motion_top200_overlap"]}条，因此名单会受指标权重影响。</p>')
parts.append('<h2>音频情况</h2><p><b>视频高分不代表发声回应已做好。</b>按现有自动检测，Top200中8条GT有声音活动、5条GT有语音；这8条在四个权重的生成结果中都没有检测到声音活动。每个权重还分别有4条在GT无检测活动的片段中检出生成声音。短回应的UTMOS仍全部N/A。</p>')
parts.append('<p>以下13条GT音轨非零，可从这里定位复核：</p><p>'+ ' · '.join(f'<a href="#pair-{r["rank"]}">#{r["rank"]} {e(r["source_id"])}</a>' for r in rs if r['gt_nonzero'])+'</p>')
parts.append('<h2>200条逐条查看</h2><p class="note">展开后可播放GT和四权重视频，独立音轨在视频下方。播放器不预加载媒体；本页需在可访问项目媒体的本地／同根目录环境打开。下方表情与声音均为自动检测结果。</p>')
for r in rs:
 label='GT非零' if r['gt_nonzero'] else 'GT全零'
 parts.append(f'<details id="pair-{r["rank"]}"><summary>#{r["rank"]:03d} · {e(r["id"])} · {r["score"]:.2f}分 · {r["duration_s"]:.2f}s · {label}</summary>')
 parts.append(f'<p>源时间 {r["source_start"]:.3f}–{r["source_end"]:.3f}s；GT运动量 {r["gt_motion_proxy"]:.4f}；自动主要表情 {e(r["gt_dominant_emotion_proxy"])}；声音活动 {r["gt_acoustic"]}，语音 {r["gt_speech"]}。</p>')
 parts.append(table(['权重','该权重排名',*labels.values()],[(names[arm],r['per_arm_rank'][arm],*[f'{r["per_arm_metrics"][arm][m]:.4f}' for m in labels]) for arm in ARMS]))
 parts.append('<div class="players">')
 for title,v,audio in [('GT',r['gt_video'],r['gt_audio'])]+[(names[arm],r['generated_videos'][arm],r['generated_audios'][arm]) for arm in ARMS]:
  parts.append(f'<div><p>{e(title)}</p><video controls preload="none" src="{link(v)}"></video><audio controls preload="none" src="{link(audio)}"></audio></div>')
 parts.append('</div></details>')
parts.append('<p class="note">这是按模型生成效果选出的高分案例集，不能代表原586条的自然难度或作为无偏测试集。没有自动冻结为正式测试名单；未人工逐条看完或听完这些样本。</p></main></html>')
(BASE/'report.html').write_text(''.join(parts))
print('Wrote report.html; verified all 2,000 referenced video/audio files exist.')
