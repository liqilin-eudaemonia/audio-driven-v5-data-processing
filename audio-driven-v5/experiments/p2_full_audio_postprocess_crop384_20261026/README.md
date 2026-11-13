# 722,706 段全量音频后处理自动接续

**已完成并通过最终验收，完成时间为 2026-10-26 20:59:13 UTC。** 本轮范围是用户确认的 listener 原始预裁框≥384×384 的 **722,706 段**，不是最初全部 1,419,124 段。静音配方、开放式整合标注、canonical 标签、新音频 latent 与事件缓存均已完成；693,997 段更新音频、28,709 段复用原音频。没有新增处理后 WAV 或复制视频缓存。依据为 [完成标记](COMPLETE.json) 与 [全量最终验收](audit.json)。

2026-10-27 再次核对：全部 722,706 个唯一 ID 与作用范围及验收 ID 一致，六类逐样本输出目录齐全；20,666 份最终验收报告的 SHA、计数和 ID 全部吻合。另抽取 32 段重新验证原音频及音频／事件缓存载荷 SHA，全部通过。721,746 段保留原逐条弱标签准入，960 段仍未准入，未自动放宽；全量正式训练未启动。

原自动顺序 **E0＋Feature → E1＋Feature → 全量音频缓存及验收** 已执行结束。以下调度与恢复说明保留作追溯，不表示当前仍在排队运行。

## 入口与进度

- [总状态](status.json)：阶段、主守护节点、错误、GPU 授权与完成数。
- [CPU 准备](producer_status.json)：归档完成数、已发布任务数与错误；处理范围为 [scope.json](scope.json) 的 722,706 个唯一 train ID、20,663 个归档。
- [配置](launch.json)：16 个 CPU 准备线程、96 个 GPU 分片，三机各 32 worker，每卡 4 worker。GPU 仅加载音频 VAE。
- [代码入口](../../scripts/p2_full_audio_postprocess.py)、[实现](../../src/listen_audio_full/cluster.py)。本轮已完成，无需再次启动处理。
- [训练接续状态](../p2_e1_feature01_train50h_20261026/queue/state.json)。E1 沿用相同 50h 数据、seed、共享初始化及 1,268 步预算，生成目标为 AV＋0.1×视频 Feature＋0.1×音频 Feature＋0.05×Event。

三机为 `172.29.252.11`、`172.29.15.91`、`172.29.225.51`。每台一个独立守护进程，共享锁确保一个主节点；其余节点定期接管检查。进程脱离终端存活，源代码和 SSH 主机公钥固定校验。

## 数据与输出

原全量根为 `../p2_features_full_crop384_20261025`，整合来源为 `../listen_full_fusion_crop384_20261025`。按 source 关联读取官方 listener metadata，沿用官方词/句/VAD 并集、边界扩展/淡变与多字中文 ASR 移除规则。

`scope.sqlite` 提供只读索引，按归档和有界任务读取；每个 worker 不再整批加载大清单。`records/`、`audits/`、`integrated/`、`view/labels/canonical/` 保留处理结果、开放原文、来源问题和训练标签。缺少有效时间的原始音频候选明确记为 unresolved，不臆造静音区间；未获原 canonical 准入的候选保留开放记录并明确排除训练。

`view/config.json` 是已通过全量验收的后处理视图，可作为后续训练适配依据；正式训练准入与子集筛选仍独立处理。复用旧 input/text/video，保存音频静音配方、新音频 latent 及 events，不新增处理后 WAV、不复制视频缓存。波形未变的样本经身份验证复用旧音频。

新视图显式关闭 Feature loss：旧 audio teacher 与修改后的音轨不匹配，不能直接用于 Feature decoder 训练。原数据 `formal_training_admitted=false`、逐条 `training_admitted` 及已有 issues 原样保留，本次不会自动改变正式训练准入。

## 自动恢复

- CPU 准备进程失败会重新拉起；已就绪任务可先被 GPU 消费，无需等全部归档完成。
- GPU worker 逐任务保存；失败不阻塞其他已就绪任务，后台重试。只清理本任务的已核验进程，不终止其他作业。
- 缓存缺失/损坏会隔离并重算，配方/整合记录缺失会请求 CPU 重建；来源或身份不符则明确报错，不混用旧缓存。
- 即使 CPU 曾完成，也可重新打开指定归档修复。worker 完成后再次审计真实载荷，最终独立审计覆盖全部 ID、原音轨 SHA、标签和缓存；已有完成标记不能替代检查。
- 最终只有全部 722,706 段通过才发布本目录 `COMPLETE.json`。GPU 进度停滞会回收本任务 worker 并重启；最终审计失败会重新进入缓存检查/修复。
- `STOP` 为人工暂停信号，不会自动删除。平时查看 `status.json`、`producer_status.json`、`workers/`、`nodes/`、`errors/`、`archive_errors/` 和 `logs/`；无需依赖当前对话持续在线。

## 已完成验证

[verification.json](verification.json) 记录源码与验证证据；[code_manifest.json](code_manifest.json) 固定 75 个文件，[code_snapshot.tar.gz](code_snapshot.tar.gz) 保存本地源码快照。冻结的 E0/E1 训练源码哈希核验一致。

- 30 项 CPU 单元/回归测试及 3 项原 50h bulk 回归通过。
- 固定 100 条真实来源与已验证 50h 结果对照：波形逐位一致，音频事件、canonical 内容一致，原准入属性保留。
- 8 条真实音频 VAE FP32 编码及训练读取通过；缺失音频重新编码逐位一致，缺失配方/整合结果/ready 可恢复。
- 31 条覆盖七类的真实 CPU 全链路（含 22 条全量启动时发现的未准入样本）：三分片生产、消费、独立最终审计通过；已有 COMPLETE 时删除载荷仍能检出，CPU 完成后重开修复成功；22 条未准入样本能够生成缓存，event known 全为 false，训练读取仍拒绝。
- [实际三机接管演练](failover_validation.json)：仅终止本任务主守护，约 6.5 秒由备用接管，跨三机检查成功，随后恢复第三个备用。

以上为启动前的 CPU 和调度验证。其后新版全量 GPU 缓存已执行完成，最终独立审计覆盖全部 722,706 段原音轨及新载荷校验，并实际抽查 104 段训练读取／未准入拒绝行为，详见 [audit.json](audit.json)。本轮只更新音频 latent 和事件缓存，未重算静音后音频的 Feature teacher，当前配置仍关闭 Feature loss。


首次全量 CPU 启动发现原通用训练校验会拒绝未准入候选，因此补充了显式“仅缓存”校验参数，默认训练校验未放宽。首次产物和错误保存在 `before_candidate_cache_fix/`；修复后独立重建视图并重新冻结源码，禁止混用两个代码版本的派生缓存。历史接管演练保留在该目录，当前版本接管记录以本目录 `failover_validation.json` 为准。
