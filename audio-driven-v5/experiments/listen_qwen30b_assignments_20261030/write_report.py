"""Render the final Markdown report from completed, audited experiment metrics."""
import json
from pathlib import Path

W=Path(__file__).resolve().parent


def main():
    m=json.loads((W/'metrics.json').read_text())
    final=m['assignments_v4'];rows=[]
    for arm,title in [('all65','65 段（此前15＋新增50）'),('repeat15','此前15段独立批次复测'),('regression8','8 个历史回归样本')]:
        x=final[arm]
        rows.append(f"| {title} | {x['accepted']}/{x['samples']} | {x['input_events_accepted']}→{x['output_events_accepted']} | {x['elapsed_s_including_model_startup']:.1f} 秒 |")
    a=final['all65'];v3=m['constrained_v3']['all65']
    p=json.loads((W/'comparison.json').read_text())['paired']['v4_same15_batch_sensitivity']
    audit=json.loads((W/'runs/regression8/audit.json').read_text())
    groups=json.loads((W/'runs/all65/audit.json').read_text())['groups']
    bad=[]
    for arm in ['all65','regression8','repeat15']:
        s=json.loads((W/'runs'/arm/'summary.json').read_text())
        bad.extend(dict(arm=arm,**x) for x in s['rejected'])
    (W/'rejections.json').write_text(json.dumps(bad,ensure_ascii=False,indent=2)+'\n')
    text=f'''# Qwen3-30B 原文整合四卡试验（2026-10-30）

**试验完成；加速收益成立，但新语义协议尚未达到全量替换条件。** 使用本机 GPU 0–3，原方案、三种中间协议及最终单归属表 v4 均保存独立响应和验收记录。没有覆盖原始打标、训练标签或缓存。测试结束后进程正常退出，无自动追加任务。

[代码、规则、参数与运行交接](../../related/P2_Qwen30B_原文整合优化与四卡验证_20261030.md) · [完整可复算统计](comparison.json) · [简表数据](metrics.json) · [失败明细](rejections.json)

## 最终 v4 结果

| 输入 | 代码＋模型自检通过 | 仅通过样本的输入→输出事件 | 耗时（含加载和重试） |
| --- | --- | --- | --- |
{chr(10).join(rows)}

65 段不是65段新人工标注：沿用此前15段复核样本，另从 node17 已完成来源抽50段，每来源一段。输入准备、随机种子及来源位于 [preparation.json](../listen_qwen30b_fusion4_20261030/preparation.json)。新增50段只有视觉＋点头输入；音频保护另在历史8段中验证。

65 段中，前一版双表协议 v3 通过 {v3['accepted']} 段，v4 通过 {a['accepted']} 段；这表示通过现有程序和模型检查的覆盖率提高，不是准确率提高的证明。最终 v4 的分组审计：

```json
{json.dumps(groups,ensure_ascii=False,indent=2)}
```

已通过输出全部完成 CPU 重放；没有 token 截断。65段通过样本中，{a['nod_observations_in_accepted_inputs']} 条原始 GLARE 点头观测均有对应来源归属。被拒样本没有删除，也未作为“已去重”输出统计。

同样15段单独组批与放入65段大批次的共同通过项为 {p['both_accepted']} 段，其中视觉输出完全相同 {p['identical_visual_outputs']} 段。temperature=0 不保证改变批次后输出逐字一致；每次运行独立保存输入和协议指纹。

## 改了什么

1. 不让模型重新抄写长描述，输出原文引用编号，由代码取原文子串。
2. 来源顺序和数量固定，不允许漏填。v4 只填一张归属表，自指行就是保留项，不再同时维护两张可能矛盾的表。
3. 将代码实际允许的代表句提前提供给模型；模型审核中补上真实起止时间和来源覆盖信息。
4. 单卡一份30B模型，batch64、32768调度token、80%显存比例，开启解码 CUDA Graph。输出长度和最多三次重试不减少。

[实测配置](tested_execution_config.json) · [v4 模块](../../src/listen_benchmark/semantic_assignment_prompt.py) · [运行入口](../../scripts/run_listener_semantic_assignments.py) · [CPU审计](../../scripts/audit_listener_semantic_prompt.py)

## 速度证据

v2 同输入 eager/graph 单次对照：15段 515.4→168.9秒；50段 622.6→349.7秒，分别约3.05倍、1.78倍。模型、token预算及重试上限相同，实际生成token接近。各组并发运行且共享主机存在其他训练，不能保证其他批次也达到这个倍数；该对照不是最终v4协议的速度承诺。

原方案 A、自由分组 v1、固定双表 v2、显式限制 v3 的全部结果均在 `comparison.json`，没有只保留最好的一次。原方案保持未修改。

## 具体成功与未解决项

- S5193：历史回归中18条视觉事件整理成11条；3条音频事件原样保留，总21→14条。重复的眼睛睁大／抬眉和向上看描述减少，保留不同时间发生的动作。可查看 [输出](runs/regression8/results/regression__naturalistic_V02_S5193_I00000190_local_60bde84ffa78.json)。
- S1400：仍拒绝。模型试图让同一长事件代表互不重叠的短来源，触发时间校验；没有强行丢掉点头或自动放行。见 `rejections.json`。
- S0220：推演后面还有可见动作，不能用“只取一个原文前缀”同时保全内容并去净推演，因此在预检查拒绝。
- 通过项也并非完全清理干净：S5193仍有 `showing continued engagement`；S0127有 `as if beginning to speak`。现有前缀识别与模型自检存在盲点，不能称为已彻底去除推演。
- 8个回归中重复yeah/okay等音频字段仍与输入一致；本轮没有重新合并音频或验证发声真假。

目前建议保留 CPU 精确去重和已验证的执行加速配置；v4 作为继续复核的候选实现，暂不推广为全量最终标签。下一步应解决原文中非末尾推演的安全处理、长状态与独立动作的区分，以及自检漏检，不能仅靠扩大并发。

## 文件与复验

- `runs/all65/`：65段主试验；`runs/repeat15/`：同15段独立批次；`runs/regression8/`：历史回归。
- 每个目录中 `protocol.json` / `code/` 冻结实际运行身份；`attempts/` 保存提案和审核原响应；`results/`、`prompts/` 仅为通过项；`summary.json` 保存全部失败；`audit.json` 为重放结果。
- `finished.json` 记录全部工作进程及 CPU 审计退出码；`run_manifest.json` 记录显卡和完整命令。
- [59项CPU测试记录](cpu_tests.json)；[20个代码文件的本地快照与SHA256](code_snapshot/SHA256.json)。这是本地备份，不是GitHub推送。
- 复算：在项目根目录运行 `python experiments/listen_qwen30b_assignments_20261030/analyze.py`，然后 `python experiments/listen_qwen30b_assignments_20261030/write_report.py`。

本报告不含人工视频复核准确率，不将模型自检当成人工真值。

## 本轮可视化

[打开本轮73段整合前后对照](http://localhost:8876/listen_benchmark_review/semantic_qwen30b_20261030/)：主试验65段＋历史回归8段，左侧有声视频、右侧原文与整合结果；未通过提案明确分开。沿用 Cursor 的8876转发端口。页面构建入口为 `scripts/build_qwen30b_assignment_review.py`，仅生成复核媒体和页面，不修改打标或训练输入。
'''
    (W/'README.md').write_text(text)
    print(W/'README.md')


if __name__=='__main__':main()
