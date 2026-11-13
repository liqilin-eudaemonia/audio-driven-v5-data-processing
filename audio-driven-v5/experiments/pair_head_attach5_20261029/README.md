# 五条来源：未覆盖短回应的句首补接试验（H 版）

在 L 版局部反馈分类与句末优先之上，增加一个可回退的句首补接步骤。仅复跑已经审核的五条来源，没有新增抽样或扫描全量语料。原生产切分、冻结音质代码与训练未修改。

[Okay 前后试听](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=okay_head#fixedPairReview) · [Yeah 前后试听](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=yeah_head#fixedPairReview) · [Right：扩展失败及回退结果](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=right_head#fixedPairReview)。原 HTML 默认 H，保留 L/G/F 历史版本；旧 comparison 和 clip 链接会选择对应历史版，不静默替换对照。

## 规则

先由原 L 版生成初始候选，随后仅对没有被任何候选覆盖（部分覆盖也不重复纳入）的短反馈尝试补接：可靠词对齐、反馈时长不超过 3 秒、无长反馈时间不确定标记，与同一人的下一段发言间隔 0～0.3 秒。0.3 秒是本次试验值，尚未证明为最佳阈值。

补接不能跨越已有候选、启用的切镜边界、质量保护区，或另一人的实质发言及其 150ms 保护区。沿用原声学函数，完整保留词，2～11.95 秒限制不变，**已有片段的终点不移动**。反馈原分类不改；已在 listener 位置覆盖的反馈不迁移。局部音质恢复子段不执行补接，避免重新扩进已排除区域。

扩展后按原 QC3/QC4 规则判定。**若扩展未通过，恢复扩展前的完整候选再按原规则判定，不因一次可选补接尝试而切碎原片段。** 扩展失败候选仍保存为待复核父片段，回退结果单独列为子段，两者不重复计入覆盖。

实现：[独立切分模块](../../scripts/listen_local_recovery_head_attach.py)。通用试验 runner 新增可选 `quality_fallback` 钩子；旧模块不提供该钩子，沿用旧行为。

## 结果

五条来源初始候选仍分别为 **20、42、19、44、55 段**。前四条初始与最终切分均不变。第五条有三个补接提案，最终采用两个：

| 回应 | 原 L 范围 | H 提案范围 | 最终处理 |
|---|---|---|---|
| Okay. | 213.983–220.346 | 213.143–220.346 | H5-28 通过；补回 Okay. |
| Yeah. | 55.070–65.450 | 54.378–65.450 | H5-08 通过；补回 Yeah. |
| Right. | 43.738–54.042 | 43.298–54.042 | H5-07 待音质复核；H5-C01 恢复原 43.738–54.042 并通过 |

第五条最终通过片段仍为 55 段，53 段起止完全不变，2 段只提前起点；通过覆盖从 306.6695 增至 308.2015 秒（增加 1.532 秒，包含前导留白）。此前已覆盖的 speaker 单词没有丢失。Right 本身仍未自动补入。

Right 扩展触发 `environmental_prediction_unresolved`，不是人工确认 Right 发音有噪声；改变片段范围也会改变音质分析窗口。页面保存扩展候选、音质证据与最终回退片段供复核。初始待音质复核父片段因此由 1→2，但原来通过的内容保留。原两条中另有一处 Oh, yeah. 满足短间隔，却受到保护边界限制，没有强行补接。

先前 San Diego／yeah、During 完整句、短尾整句与 up／Oh, yeah. 修复均保留。During 的末尾零留白提示，以及第三条 it’s 附近的重叠发声提示，也仍然存在。

## 验证与运行

- 29 项恢复测试通过，包括真实五来源、部分已覆盖事件、另一人插话、质量保护区、切镜、不可靠对齐、超间隔、长度约束和扩展质量失败回退。
- [逐来源比较](comparison.json) · [完整候选及媒体清单](review/head_attach/manifest.json) · [数据验收](validation.json) · [浏览器验收](browser_validation.json)。
- 新版共 197 条父／子候选媒体记录，其中 195 条精确复用已验证同边界媒体，2 条新导出并完整解码检查。浏览器串行播放 17 个新旧对照视频，验证音轨切换、历史链接、反馈版本隔离和手机布局。
- 第一次探索发现 Right 扩展失败后若直接走局部恢复，会丢掉原来通过的句尾；该结果仅存于 `results_initial/`，不作为默认展示结果。最终 `results/` 已应用完整回退，所有旧覆盖词保留。新扩展音质证据已计算，最终运行精确复用证据后重新路由；日志保留。
- 按用户要求，后续媒体导出、发布和浏览器验收采用单任务串行。执行进程沿用前次验证的 THP 禁用措施，未修改宿主机配置。

项目根目录执行：

```bash
python experiments/pair_head_attach5_20261029/run.py
python experiments/pair_head_attach5_20261029/export.py
python experiments/pair_boundary_real_audit_20261029/publish_review.py
python -m unittest discover -s tests -p 'test_listen_local_recovery*.py' -v
PYTHONPATH=/tmp/v8_browser_deps LD_LIBRARY_PATH=/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu python experiments/pair_head_attach5_20261029/check_browser.py
```
