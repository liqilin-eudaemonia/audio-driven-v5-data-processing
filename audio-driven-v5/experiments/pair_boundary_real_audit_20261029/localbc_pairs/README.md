# 两条来源：局部反馈分类与句末优先试验

本版 `local-feedback-sentence-priority-4`（HTML 编号 L）只重切原有 190 秒与 353.987 秒来源；G 为之前的 5 秒版，F 为 3 秒版。原生产结果、冻结生产代码和训练未修改。

[San Diego／yeah 前后试听](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=local_feedback#fixedPairReview) · [During 整句前后试听](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=sentence_priority#fixedPairReview) · [第二条相邻切点变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&compare=sentence_priority_2a#fixedPairReview)。均在原 HTML 中，左右实际片段无额外上下文，支持切换历史版本与记录人工结论。

## 两处修改

1. 先以可靠词对齐和 VAD 双重确认的超过 5 秒停顿，将过粗转写段拆成局部发声，再沿用既有反馈分类器判断每一段。第一条 A 的长转写末尾 `yeah` 因而独立成为反馈候选，不再继承之前实质发言的阻断。仅这一处转写段发生拆分。保留 VAD 分区全部活动，避免把最后一个词之后仍有的实质声音错误放行；无可靠对齐或 VAD 不支持时不拆分。
2. 贪心切分和短尾回退时，真正句末优先于推测分句末，再比较原有时长偏好，避免逗号分句因为短于 10.8 秒而抢先于可放下的完整句。

5 秒相邻词长停顿阈值、2～11.95 秒限制、实质发言的 150ms 安全余量、声学切点函数、QC3/QC4 及准入规则均沿用。没有针对频繁打断的 G2-27 或源视频末尾录制不全添加特例。独立试验模块为 [listen_local_recovery_localbc.py](../../../scripts/listen_local_recovery_localbc.py)。

## 结果

| 来源 | 初始候选 G→L | 边界完全相同 | 音质待复核 | 语义边界提示 G→L | 局部恢复后通过覆盖 G→L |
|---|---:|---:|---:|---:|---:|
| V02_S3579_I00000175 | 21→20 | 18 | 4→4 | 11→10 | 140.236→142.667 秒 |
| V03_S1909_I00000205 | 42→42 | 38 | 6→6 | 20→18 | 220.336→219.928 秒 |

音质待复核数包括规则暂缓与规则排除的初始父 pair，不能和恢复子段重复计数；语义提示与音质疑点可重叠，也不等于已确认截断。

- L1-01：5.743–11.700 秒，共 5.957 秒。保留 `San Diego?`、listener `yeah` 与说话者随后 `Okay.`；现有音质规则通过。
- L1-16：149.903–161.853 秒，共 11.95 秒。原 G1-16/G1-17 合为从 `During the time I see the therapist,` 到 `marry her.` 的整句；现有音质规则通过。**句尾额外留白为 0，仍有未找到安静切口、尾部余量不足提示，需重点试听结尾。** 不能把转写覆盖完整当成声学收尾已由人工确认。
- 第二条 G2-09/10、G2-14/15 的相邻切点调整，全部在页面另附左右对照。原先已覆盖的 speaker 词没有丢失，覆盖时长小幅减少不能等同于丢词。
- 早先短尾整句遗漏、`up.` 与 `Oh, yeah.` 遗漏两处修复仍保留；`It was crazy` 仍在同段。

共 62 个初始候选＋7 个局部恢复子段，69 次质量判定：6 段新边界重算 QC3/QC4，63 段边界与归一化音轨完全相同，复用质量证据后重新路由；全部媒体解码校验完成。23 项恢复测试通过；[数据与冻结文件校验](validation.json)、[完整清单](../review/localbc_pairs/manifest.json)、[浏览器验收](../localbc_browser_validation.json)。

## 复现

在项目根目录运行（音质运行使用现有项目依赖环境）：

```bash
python experiments/pair_boundary_real_audit_20261029/rerun_localbc_pairs.py
python experiments/pair_boundary_real_audit_20261029/export_localbc_pairs.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_boundary_real_audit_20261029/check_localbc_pairs.py
```

运行入口通过共享 runner 的可选 `prepare_people` 钩子应用局部分类；既有 F/G 模块没有此钩子。原始词时间、VAD 和来源 JSON 不覆盖。实验合同见 [contract.json](contract.json)，重跑及导出日志位于本目录上一级。
