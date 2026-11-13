"""Add computed paired metrics without changing the default speaker mix."""
from pathlib import Path
import html,shutil,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=HERE/'preview_metrics';OUT=ROOT/'experiments/p2_av_review_20261027/autonomous_preview_20261028'
COLS=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID']
def publish():
 result=json.loads((WORK/'comparison.json').read_text());assert result['status']=='complete'
 def table(mode):
  item=result['modes'][mode];body=''
  for row in item['rows']:
   cells=''.join('<td>'+('N/A' if row['values'][k] is None else f'{row["values"][k]:.4f}')+'</td>' for k in COLS)
   body+='<tr><td>'+html.escape(row['name'])+'</td><td>6</td>'+cells+'</tr>'
  return '<div class="table-scroll"><table><thead><tr><th>权重／指导方式</th><th>输出数</th>'+''.join('<th>'+k+(' ↑' if k in ['UTMOSv2','LSE-C','R-F1','R-tIoU'] else ' ↓')+'</th>' for k in COLS)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
 c=result['modes']['unfiltered']['coverage'];n=result['modes']['strict']['coverage']['glare']
 section='<section class="metrics-panel"><h2>3条样本 · 配对评测结果</h2><p>已完成12个自主规划输出的计算，并取同两组权重、同3条样本、同seed137/251的真实标注指导结果重新汇总。每条件6个输出；各seed分别计算后等权平均。<a href="metrics_report.md" target="_blank">完整报告</a> · <a href="metrics_comparison.json" target="_blank">逐项结果与覆盖率</a></p>'
 section+=f'<p class="metric-coverage">四条件＋双seed共同支持：FVD {c["fvd"]}/3 · UTMOS {c["utmos"]}/3 · SyncNet {c["sync"]}/3 · GLARE去额外门槛 {c["glare"]}/3。</p>'
 section+='<p class="metric-warning">UTMOS/SyncNet已运行，配对表N/A表示没有共同有效样本：2条GT为静音，另1条的自主规划AV输出在两个seed下均被判为静音。可用的逐视频分数仍显示在下方卡片及JSON中；不能将各条件不同有效集合直接比较。评测使用独立listener.wav，不使用speaker叠加音轨。</p>'
 section+='<h3>去额外门槛版 · 全3条</h3>'+table('unfiltered')+f'<details><summary>严格逐帧版 · 共同支持{n}/3条</summary>'+table('strict')+'</details>'
 section+='<p class="note">仅3/24条规划成功样本，存在选择偏差；FVD/R-FID在这么小的样本上不稳定。AV的自主规划视觉指标改善，但AV＋Event多数视觉反应指标变差，没有一致优势。这3条规划均无发声安排，不能据此认定自主回应能力已经通过验证。</p></section>'
 p=OUT/'index.html';s=p.read_text();a=s.index('<section class="metrics-panel">');b=s.index('</section>',a)+len('</section>');s=s[:a]+section+s[b:];s=s.replace('仅3/24条规划通过校验；是成功样本预览，不是完整评测结果。','仅3/24条规划通过；本页已完成这3条的配对计分，不是完整test24结果。');p.write_text(s)
 p=OUT/'data.json';d=json.loads(p.read_text());d['metrics_computed']=True;d['paired_metrics']=result['modes']
 metrics={a:{(r['sample_id'],r['sample_seed']):r for r in json.loads((WORK/a/'subset_results.json').read_text())['rows']} for a in ['auto_av','auto_av_event']}
 for sample in d['datasets'][0]['samples']:
  for output in sample['outputs']:
   arm='auto_av_event' if 'av_event' in output['arm'] else 'auto_av';r=metrics[arm][sample['id'],output['seed']]['metrics'];output['metrics'].update(utmos=r['utmos']['generated'].get('value'),lse_d=r['sync']['generated'].get('lse_d'),lse_c=r['sync']['generated'].get('lse_c'));output['metric_status']='evaluated';output['metric_details']=r
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 shutil.copy2(WORK/'report.md',OUT/'metrics_report.md');shutil.copy2(WORK/'comparison.json',OUT/'metrics_comparison.json')
 p=OUT/'publication.json';v=json.loads(p.read_text());v.update(metrics_computed=True,comparison_conditions=4,metric_coverage=c,strict_glare_common=n);p.write_text(json.dumps(v,indent=2)+'\n')
 p=ROOT/'experiments/p2_av_review_20261027/index.html';s=p.read_text().replace('3条通过校验的样本，AV/AV＋Event、双seed共12个视频；尚未计分。','3条通过校验的样本，AV/AV＋Event、双seed共12个视频；已完成同样本配对计分。');p.write_text(s)
 print('Published paired metrics,12 output cards,coverage',c)
if __name__=='__main__':publish()
