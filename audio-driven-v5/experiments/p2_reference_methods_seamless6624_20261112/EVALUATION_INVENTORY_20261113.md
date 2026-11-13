# 本轮参考方法已测统计

范围：Learning2Listen四套、ListenFormer、DyStream、Avatar Forcing、DualTalk，共5个方法、8个配置，其中7个RGB配置、1个仅原生动作配置。只统计这轮参考方法；不将训练接入短跑或自有模型权重网格计入。

所有node11全量任务已暂停，自动恢复关闭。暂停快照：2026-11-13 14:57:14 UTC+8。下列全量数量重新扫描完成回执核对，未启动GPU或恢复任务。

## 已完成评测范围

| 数据集／范围 | 方法配置覆盖 | 结果状态 |
|---|---|---|
| ViCo正式test43 | 全部8个配置，各43条；DualTalk仅原生动作 | 已完成对应指标；图像7行，DualTalk原生成绩单列 |
| ResponseNet历史固定200 | 全部8个配置；L2L每套199个视频＋1条N/A | 推理及对应指标完成 |
| Seamless历史固定200 | 全部8个配置；L2L每套198个视频＋2条N/A | 推理及对应指标完成 |

双200共有24份最终结果JSON：14份视频结果＋10份L2L／DualTalk原生动作诊断结果。指标有支持不足／官方追踪无身份时保留partial／N/A，不补零；任务完成不表示每项对每条都有有效分数。

## Seamless完整6624：暂停时实际产物

| 方法 | 原生动作 | 已生成视频 | 视频N/A | 已评分请求 |
|---|---:|---:|---:|---:|
| Avatar Forcing | — | 1283 | 0 | 1200 |
| DyStream | — | 631 | 0 | 300 |
| ListenFormer | 987 | 985 | 0 | 200 |
| L2L Conan | 975 | 386 | 2 | 200 |
| L2L Fallon | 975 | 386 | 2 | 200 |
| L2L Stephen | 975 | 386 | 2 | 200 |
| L2L Trevor | 975 | 386 | 2 | 200 |
| DualTalk | 988 | N/A | — | 待全量动作汇总 |

合计4443个有效生成视频，评分分片覆盖2500条请求（含L2L共8条生成N/A）。尚无完整6624最终汇总结果。历史200已包含在这些数量内，不能再相加。

共用缓存：speaker输入特征988条、评测GT特征1311条、GPAvatar首帧身份863条，另2条无身份；缓存数量不等于模型推理或评分完成数量。

## ResponseNet完整586

新增386条尚未推理；现有缓存来自历史200条：AF／DyStream／ListenFormer各200个视频，L2L四套各199个视频＋1条N/A，DualTalk及其他动作生成模型各200条动作。全量586尚未完成。

本轮可复用评分为AF／ListenFormer／L2L各200条；DyStream因评分指纹不兼容需重评，不能据此说历史DyStream200没有测过。

## 视频产物去重合计

ViCo主测试301份＋ResponseNet1396份＋Seamless4443份＝6140份有效生成视频。不同方法／权重对同一输入的输出分别计数；历史固定200只计一次，不重复累加全量缓存、分辨率副本或AF专项复查。DualTalk原生动作不计为视频。

## 结果入口

- [ViCo完整结果及Report对照](../p2_vico_full_eval_20261111/README.md)；[AF追加完整结果](../p2_vico_avatarforcing_added_20261111/README.md)。
- [ResponseNet200＋Seamless200最终数值表](../p2_reference_methods_dualtest_20261112/RESULTS.md)。
- [Seamless全量阶段状态](status.json)；[ResponseNet全量阶段状态](../p2_reference_methods_responsenet586_20261112/status.json)。
- [本次机器可读统计](EVALUATION_INVENTORY_20261113.json)。
