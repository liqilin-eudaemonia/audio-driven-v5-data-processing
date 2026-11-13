#!/usr/bin/env python3
"""Require complete, version-separated cohorts before exporting the diagnostic report."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json
from listen_benchmark.runner import select_samples, review_template
from listen_benchmark.evaluate import subtract
from listen_labeling.schema import CLASSES


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run).resolve()
    cfg=json.loads((root/'code_v2/configs/listen_benchmark.json').read_text())
    latest=json.loads((root/'comparison_latest.json').read_text());lookup={(r['phase'],r['system']):r for r in latest['rows']}
    selected=json.loads((root/'selected_phases.json').read_text()) if (root/'selected_phases.json').exists() else {}
    chosen=[];missing=[]
    for system in cfg['systems']:
        name=system['name'];phase='transformers_v2'
        if name in ['joy_visual','qwen3_visual','qwen3_audio','qwen3_omni_separate','qwen3_vl_visual']:phase='vllm_v3'
        if name in ['minicpm_omni_native','minicpm_omni_native_1fps']:phase='tail_fixed_v4'
        if name=='captioner_audio':phase='captioner_template_v6'
        if name=='qwen3_omni_native':phase='native_vllm_v5'
        phase=selected.get(name,phase)
        row=lookup.get((phase,name))
        if not row or row['completed_samples']!=row['expected_samples']:missing.append((phase,name,None if row is None else row['completed_samples']));continue
        chosen.append(dict(row,tier=system['tier'],auxiliary=system['auxiliary']))
    if missing:print(json.dumps(dict(pending=missing),ensure_ascii=False));return 2
    samples=select_samples(root/'manifest50.json',None);ids={s['sample_id'] for s in samples};duration=sum(s['duration_s'] for s in samples)
    passed_ids={};covered_ids={}
    for row in chosen:
        summary=json.loads(Path(row['summary']).read_text());identity=json.loads(Path(row['summary']).with_name('inputs.json').read_text())
        if {s['sample_id'] for s in identity['samples']}!=ids:raise ValueError('Cohort mismatch')
        if any(s['media_hashes']!=next(v['media_hashes'] for v in samples if v['sample_id']==s['sample_id']) for s in identity['samples']):raise ValueError('Media identity mismatch')
        stamps=[datetime.fromisoformat(v['utc']) for v in summary['results']]
        ends=[datetime.fromisoformat(v['utc'])+timedelta(seconds=v.get('wall_seconds_including_preparation_and_load',0)) for v in summary['results']]
        row['observed_request_span_seconds']=(max(ends)-min(stamps)).total_seconds()
        row['audit_supported_seconds_fraction']={k:v/duration for k,v in row['checked_support_seconds'].items()}
        if not row['auxiliary']:
            items={}
            for result in summary['results']:items.setdefault(result['sample_id'],{})[result['stage']]=result
            passed_ids[row['system']]={sid for sid,stages in items.items() if all(v['status']=='schema_pass' for v in stages.values())}
            supported=set()
            for sid,stages in items.items():
                audit=stages['audit']
                if audit['status']!='schema_pass':continue
                value=audit['parsed'];durations=[]
                for label in CLASSES[summary['branch']]:
                    excluded=value['unknown'][label]+[[e['start'],e['end']] for e in value['events'] if e['type']==label and e['boundary_censored']]
                    durations.append(sum(b-a for a,b in subtract(value['reviewed'][label],excluded)))
                if all(x>0 for x in durations):supported.add(sid)
            covered_ids[row['system']]=supported
    pairs=[]
    for visual,audio in [('joy_visual','qwen3_audio'),('qwen3_vl_visual','qwen3_audio'),('qwen3_visual','qwen3_audio'),('minicpm_visual','minicpm_audio')]:
        pairs.append(dict(scheme=visual+' + '+audio,configurations=[visual,audio],four_stage_schema_pass=len(passed_ids[visual]&passed_ids[audio]),
                          both_targets_nonzero_coverage=len(covered_ids[visual]&covered_ids[audio]),expected_samples=len(samples)))
    atomic_json(root/'paired_scheme_summary.json',dict(pairs=pairs,human_ground_truth=False,training_admitted=False))
    report=dict(utc=datetime.now(timezone.utc).isoformat(),samples=len(samples),duration_seconds=duration,configuration_count=len(chosen),
                completed_config_samples=sum(v['completed_samples'] for v in chosen),completed_requests=sum(v['completed_requests'] for v in chosen),
                human_ground_truth=False,training_admitted=False,rows=chosen)
    records=[json.loads(s) for s in (root/'cluster_resources.jsonl').read_text().splitlines()]
    resources={}
    for node in ('node17','node12','node4'):
        cpu=[r['nodes'][node]['cpu_percent'] for r in records if r['nodes'][node].get('cpu_percent') is not None]
        gpu=[g for r in records for g in r['nodes'][node].get('gpus',[])]
        if not cpu or not gpu:raise ValueError('Missing resource telemetry: '+node)
        resources[node]=dict(cpu_mean_percent=statistics.mean(cpu),cpu_peak_percent=max(cpu),
                             gpu_mean_percent=statistics.mean(g['gpu_percent'] for g in gpu),
                             gpu_peak_percent=max(g['gpu_percent'] for g in gpu),
                             peak_vram_gib_per_gpu=max(g['memory_used_mib'] for g in gpu)/1024,
                             observed_gpu_indices=sorted({int(g['index']) for g in gpu if g['memory_used_mib']>1000}))
    atomic_json(root/'resource_summary.json',dict(interval=[records[0]['utc'],records[-1]['utc']],
                includes_loading_and_idle_tail=True,nodes=resources))
    atomic_json(root/'final_comparison.json',report);atomic_json(root/'blind_review50.json',review_template(samples))
    lines=['# 三机打标工具首轮诊断结果','',f"完成：{report['utc']}。同一批 {len(samples)} 条 listener 样本、{len(chosen)} 种配置，共 {report['completed_requests']} 次最终纳入的请求。",'',
           '**这是运行、输出结构和模型自报覆盖的诊断；尚无人工真值，不能报告准确率或宣布最终最佳模型。** 失败请求保留在分母内，所有结果仍是待审草稿。','',
           '| 配置 | 实际执行版本 | 完成 | 两阶段结构通过 / 辅助非空输出 | audit 两类/单类均有有效覆盖 |',
           '|---|---|---:|---:|---:|']
    for row in chosen:
        count=(row['auxiliary_completed']-row.get('empty_auxiliary_outputs',0)) if row['auxiliary'] else row['complete_schema_pass']
        lines.append(f"| {row['system']} | {row['phase']} | {row['completed_samples']}/50 | {count}/50 | {'—' if row['auxiliary'] else str(row['audit_supported_all_target_classes'])+'/50'} |")
    lines+=['','“结构通过”包含 JSON、时间范围、事件与 reviewed 区间一致性等校验；“有效覆盖”指各目标均有非零的 reviewed 时长，并扣除了 unknown 和模型自报的边界截断区间，不表示整段已覆盖。这两个指标都不能证明事件判断正确。','',
            '| 分路组合 | 四阶段全部结构通过 | 两个目标均有非零有效覆盖 |','|---|---:|---:|']
    for pair in pairs:lines.append(f"| {pair['scheme']} | {pair['four_stage_schema_pass']}/50 | {pair['both_targets_nonzero_coverage']}/50 |")
    lines+=[
            '', '## 运行与修正', '',
            '- 三台机器各 8 张 H200（143771 MiB/卡）、160 核 CPU；使用共享样本队列、进程锁、常驻模型和多副本/批量推理。任务结束即释放显卡。',
            '- Transformers 首轮与 vLLM 批量版分别留档，不拼成一个虚构的同版本成绩。视觉/音频 Instruct 分支保留 4096 token 上限；原生视听、普通拼接和 1 fps 控制分别记录。',
            '- MiniCPM 片尾 WAV 比最后视频帧稍长时，初版报缺帧；v4 保留真实尾音频、不复制帧，重新执行两种原生视听配置的全 50 条。旧失败保留。',
            '- Captioner 初版额外插入关闭思考的标记，50 条中 21 条空描述。v6 改为官方默认音频模板并在 vLLM 重跑，空描述现在明确记为失败；旧版本不纳入最终辅助结果。',
            '- Qwen 原生视听在 vLLM 0.19 混合/分块预填充中出现 embedding 形状错误；单请求、关闭分块的 v5 先通过两阶段冒烟，再独立执行全 50 条。本表采用 v5 完整组；Transformers 的已完成结果及长尾请求另存。',
            '- 单机 CPU/GPU 利用率、显存和功率实测见 `cluster_resources.jsonl`，统计见 `resource_summary.json`；更早的记录见 `pilot/*/resources.jsonl`、`full50_v2/nodes/*/resources.jsonl`。模型加载、串行解码和收尾阶段存在利用率下降，未声称全程 100%。',
            '', '## 复核入口', '',
            '- `final_comparison.json`：最终选择的版本、结构统计、有效覆盖秒数及请求时间跨度。不同并发规模的时间跨度不能作为模型固有速度排名。',
            '- `comparisons/<version>/<system>/summary.json`：完整逐样本结果；每条记录链接原始请求、原始文本、token、输入证据及失败原因。',
            '- `failed_requests.json`：所有阶段的失败，含已替代旧版本；筛选时须带 phase。',
            '- `blind_review50.json`：不含模型答案的人工复核模板。当前 50 条属于开发诊断样本；正式选型还需按人物/session 隔离的新留出集。',
            '', 'Captioner 输入方式依据 [官方模型说明](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)。实际下载位置为 `/public/2_CKPT/3_PreTRAINED`，不需要再下载。']
    if (root/'retest_plan.json').exists():
        # Keep the computed result tables; replace first-run history with this run's provenance.
        lines=lines[:lines.index('## 运行与修正')]
        lines[0]='# 三机打标工具完整重测结果'
        lines += ['## 本轮重测', '',
                  '- 同一批 50 条样本、16 种配置、1450 次请求全部重新执行；未复用上一轮标签。',
                  '- Qwen3-Omni 原生视听显式传入 second_per_grid_ts=[0.25]，修正 8 fps 视频在 vLLM 缓存路径中的 token 交错时序。保留单请求、关闭分块预填充设置。',
                  '- 样本、prompt、生成预算和各配置推理引擎保持原轮口径；MiniCPM 尾段修正和 Captioner 默认模板继续保留。',
                  '- 三节点 24 张 H200 执行；模型加载与收尾的利用率实测见 cluster_resources.jsonl。显存占用不等同于计算利用率。',
                  '- 旧实验路径及变更范围见 retest_plan.json。没有人工事件真值，新增事件不等于召回改善。',
                  '- comparisons/ 保留全部请求、输出、失败与配置指纹。所有标签仍待人工复核。']
    (root/'最终对照结果.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
