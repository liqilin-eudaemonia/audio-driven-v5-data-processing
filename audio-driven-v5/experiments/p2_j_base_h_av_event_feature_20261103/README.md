**当前续训已迁移三机24卡：** [配置、迁移审计及实时状态](../p2_j_base_h_av_event_feature_continue24_20261105/README.md)。下方为原28卡运行历史。

**2026-11-05已排队恢复：** 等当前Whisper实验完整保存第8000步并退出后，严格从本实验5500步恢复；保持四机28rank/global28及原全部训练配置。[接续状态与验证](../p2_whisper8000_to_event_feature_20261105/README.md)。

# 原始H＋AV＋Event＋Feature：四机28卡

2026-11-03按用户要求训练组合监督。初始请求32卡，实查node4的4–7卡在运行Lingbot任务，用户明确改用剩余28卡；没有停止其他任务。

## 数据与损失

当前500h、285638条train，原始未规划SFT的Qwen2.5-Omni-3B guided H，原始完整打标描述，最后修正版listener 512×512裁框、官方时间并集净化listener音频。继续使用原缓存和已验收独立teacher，不重新提取。Event为现有16视觉＋3音频clip多标签监督，不改类别或负例权重。

`L = L_AV + 0.05*L_Event + 0.025*L_video_feature + 0.025*L_audio_feature`。

Qwen冻结、H缓存读取；Connector接收三种损失，Event head由Event更新，Feature decoder由Feature更新，LTX text-attention LoRA由AV更新。Feature和Event不直接更新LTX LoRA。新初始化训练，不从已完成的Feature或Event权重续训。

## 配置与资源

- 实际配置：`configs/p2_j_base_h_av_event_feature_28gpu_20261103.json`。
- 集群：`configs/p2_j_base_h_av_event_feature_cluster28_20261103.json`。
- node17/node8/node12各8卡；node4仅0–3卡。torchrun异构本地进程数8/8/8/4，总world28，显式关闭同构rank快速假设。
- 每卡1条，global batch28，完整一轮ceil(285638/28)=10202步。LR、seed、Connector1024/4层/16头与既有组一致。
- 从第1步duration分组：448样本窗口＝16步，保留所有样本与确定性序号。BF16、three_quarters梯度重算、32MB梯度桶及既有网络配置。
- 每500步保存，第6000步自然保存，最终第10202步另存。初始GPU验收额外保存0和16步；前16步计入正式预算，完整模型/优化器/28rank RNG验收后严格续跑。
- 原32卡配置文件只作准备记录，没有启动。当前训练以28卡配置为准。

batch32改为28、duration启动点改变，不能宣称与旧组是完全相同优化轨迹的单因素实验；6000步分别消耗168000与192000条，需结合完整一轮结果比较。数学目标中的三项权重没有为了卡数缩小而更改。

## 代码、验收和恢复

现有MatrixModel与clip backend本身已支持av_event_feature，无需修改冻结缓存或原训练核心。新增 `scripts/p2_j_event_feature_launch.py`：资源核查、复用同源码/同数据身份的全量缓存验收、28卡16步反传/断点验证、自动接续。`tests/test_p2_j_event_feature.py`验证权重与梯度路径。

45项CPU回归通过。全量285638条preflight复用已完成Feature组的有效全量审计：数据、Event标签、teacher、源码SHA完全一致，记录在preflight.json的reused_full_preflight。初次重复扫描已停止，仅停止本控制器CPU进程组，partial日志保留；不等待重复扫盘再启动GPU。

真实3条缓存读取与H/teacher/Event组合检查见 `CPU_VALIDATION.json`。GPU验收必须看 `GPU_VALIDATION.json`，出现前不能宣称完成反传验收。状态 `status.json`、`queue/state.json`、日志 `controller.log` 及 `queue/initial16/`、`queue/continue_epoch/`。控制器PID在 `controller.pid`；不要重复启动。

启动入口（只在旧控制器已退出时使用）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_j_event_feature_launch.py \
  --config configs/p2_j_base_h_av_event_feature_28gpu_20261103.json \
  --cluster configs/p2_j_base_h_av_event_feature_cluster28_20261103.json
```

故障恢复须用本实验最新 `.complete.json` 配对且SHA有效的checkpoint，通过 `launch_commands.json` continuation命令替换resume路径并使用新日志目录。启动600秒无指标、训练中180秒无新指标会取消仅本控制器所属进程，写queue/STALL.json，不跳过样本或伪报完成。完成以step10202/complete=true及最终checkpoint为准。

## 28卡网络恢复记录

初次8/8/8/4运行已形成完整step0，但更新前出现RoCE跨rail错误：node4↔node12/17 completion status12/5及unable to allocate requests。只停止本实验guard所属进程，保留step0、原日志和network_recovery.json；其他Lingbot任务保持运行。统一使用 `mlx5_0:1`、GID3，从同一step0严格恢复；不修改模型、优化器、数据或数学损失。此处多网卡旧32卡配置不能直接作为28卡异构通信的验收结论。实际通信性能和GPU成功以GPU_VALIDATION.json为准。

## GPU启动验收已完成

28rank首16步反传及完整checkpoint通过，已严格恢复并继续超过第16步；首个窗口平均4.86秒/步，rank0峰值预留显存92.73GiB。28卡资源采样在GPU_STARTUP_SNAPSHOT.json，node4未选4–7卡仍为Lingbot任务。READY.json记录CPU/缓存/GPU/续训状态。当前源码和配置快照在source_snapshot/，附SHA256清单。
