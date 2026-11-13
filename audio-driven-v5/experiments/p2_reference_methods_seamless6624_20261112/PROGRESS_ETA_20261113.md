**耗时估计已更新：[最新复核](ETA_RECHECK_20261113.md)。以下保留原快照。**

# 参考方法全量进度和预计时间

快照：2026-11-13 11:04:52 UTC+8。仅node11 GPU7；11个常驻阶段正常运行。历史ResponseNet200＋Seamless200已全部完成。

## Seamless 完整6624

| 方法 | 动作 | 视频 | 视频N/A | 已评分请求 |
|---|---:|---:|---:|---:|
| DyStream | — | 333 | 0 | 300 |
| Avatar Forcing | — | 552 | 0 | 50 |
| ListenFormer | 594 | 512 | 0 | 200 |
| L2L Conan | 575 | 292 | 2 | 200 |
| L2L Fallon | 575 | 292 | 2 | 200 |
| L2L Stephen | 575 | 292 | 2 | 200 |
| L2L Trevor | 575 | 291 | 2 | 200 |
| DualTalk | 593 | — | — | — |

## ResponseNet 完整586

| 方法 | 动作 | 视频 | 视频N/A | 已评分请求 |
|---|---:|---:|---:|---:|
| DyStream | — | 200 | 0 | 0 |
| Avatar Forcing | — | 200 | 0 | 200 |
| ListenFormer | 200 | 200 | 0 | 200 |
| L2L Conan | 200 | 199 | 1 | 200 |
| L2L Fallon | 200 | 199 | 1 | 200 |
| L2L Stephen | 200 | 199 | 1 | 200 |
| L2L Trevor | 200 | 199 | 1 | 200 |
| DualTalk | 200 | — | — | — |

Seamless继续新增推理与评分；ResponseNet仅复用历史200条，新增386条等待Seamless全量完成。评分列只计本轮已验收或指纹匹配可复用的分片。旧200条评过但评分指纹不兼容的分片需要重评；不能将本轮0解读为历史没有测试。

## 预计剩余时间

基于当前并行负载下最近30个片段的耗时，并按剩余音视频时长归一化；评分依据3个正式25条分片估计。不是把历史等待时间计入模型速度。模型保持常驻，未触发重新加载。

| 方法／阶段 | Seamless推理剩余粗估 |
|---|---|
| Avatar Forcing | 约1.5–2天 |
| DyStream | 约3.5–4.5天 |
| ListenFormer | 约3–4天，受共用speaker特征约82小时限制 |
| DualTalk | 约3–4天，受共用speaker特征约82小时限制 |
| L2L四套完整视频 | 约10–13天，四套共用GPAvatar渲染；当前外推约271小时 |

七组RGB视频的剩余评分合计外推约215小时，与生成并行推进，不与生成时间直接相加。Seamless完整结果按当前分配保守预计10–13天，约北京时间11月23–26日。ResponseNet586接续后另约1–2天，两数据集全部结束约11月24–28日。

这些是当前单卡吞吐的条件外推：其他阶段完成后资源释放可能提速；若GPU7被前台占用，让位时间会延长。最终完成仍以各目录COMPLETE.json为准。

原始计时与计算：[阶段计时](ETA_TIMINGS_20261113_0302.json) · [计算快照](PROGRESS_ETA_20261113.json)。实时数量：[Seamless状态](status.json) · [ResponseNet状态](../p2_reference_methods_responsenet586_20261112/status.json)。
