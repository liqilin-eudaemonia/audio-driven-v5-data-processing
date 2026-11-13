**2026-11-09再次复核通过：** 修复CLI同时传入不一致arm／config时未拒绝的入口保护缺口。30项测试通过，包含新增两进程真实Event归一化／Adam对照、全重计算loss及全部梯度逐值一致、Whisper音视频影响／padding mask；两组重跑各32条canary，另各16条不重复样本走真实后台预取，完整caption及Event数值标签一致。295份当前训练源码保持原样。冷入口和preflight身份已重新生成，备份同步。实际GPU多卡验收仍待正式启动，不能据此承诺GPU显存或NCCL无故障。[复核验收](REVIEW_ACCEPTED.json)。

# 全LTX LoRA：去Feature、去Whisper

2026-11-09，补齐当前完整组的两个逐组件消融。状态：代码／CPU原生模型／完整模型meta覆盖／真实缓存预检已完成；没有启动GPU训练，也没有加入自动队列。实际完整规模GPU显存、NCCL和步速需要在分配训练资源后的启动检查中确认。

| arm | 条件 | 损失 | 完全移除 |
|---|---|---|---|
| `no_feature` | 原始Qwen hidden＋speaker Whisper | AV＋0.05整段Event | 两路EmbeddingDecoder、Feature优化器组、teacher读取 |
| `no_whisper` | 原始Qwen hidden | AV＋0.05整段Event＋0.025视频Feature＋0.025音频Feature | Whisper adapter、两路Whisper cross-attention、Whisper优化器组和缓存读取 |

去Whisper仍通过Qwen条件携带speaker音频信息。去Feature保留新增 `qwen_attn2`、`audio_qwen_attn2` 全参数训练。两组原生LTX的self-attention、条件attention、音视频交叉attention、实际attention gate及FFN均使用rank8／alpha8 LoRA：48层1632处，84,221,952参数；原生基座冻结。去Feature的Whisper attention参数2,779,450,368；去Whisper为0。Connector及启用的辅助头全参数训练。Qwen读取原始冻结hidden缓存，不是在线Qwen联合微调。

Event读取实际LTX最后block的hidden，使用整段聚合；19类配置、18类启用，mouth closing沿用原5000门槛屏蔽，无逐帧／顺序损失。Feature沿用EmbeddingDecoder监督Connector输出，不更新冻结Qwen。

## 数据和可比性

两组严格引用旧组件版本的最终500h配置，只扩展原生LTX LoRA范围。最终train＋dev284,218段／500.001259h，train281,060＋dev3,158，无test；完整最新caption、原始Qwen H、Gemma／AV缓存及启用的teacher／Whisper缓存复用；当前数值Event标签和source digest均保持。各32条真实回读及完整caption一致性检查通过，既有全量缓存／标签验收绑定不变。

从同一预训练基座及相同seed独立初始化，不使用已训练完整模型删组件后续训。共享参数初始化、AV和保留辅助项前向与完整组逐值一致。

本轮准备沿用完整组6000之前的batch31／duration496／原sample-noise计划（9169步一轮、尾部21条padding），以便6000步对齐6000×31次样本展示。默认保存每500步。四机31卡配置沿用node12／node8／node4各8卡，node11 GPU1–7、GPU0预留；这只是未启动模板，不改变当前28卡训练及node11 GPU4–7预留。若后续使用28卡，应另建经验证的batch28配置并按累计样本量对齐，不能直接改当前冻结配置或强行启动31卡模板。

保留全重计算、CPU prefetch2、GPU Adam、512MiB streaming FP64归约和四机已验证的单网卡配置。没有采用已实测OOM的减重计算策略。

## 代码、入口及恢复

- 独立包：`src/p2_ltx_all_lora_components/`。
- 入口：`scripts/p2_ltx_all_lora_components_train.py`。
- 两组配置及预检：本目录 `no_feature/`、`no_whisper/`。
- 独立checkpoint schema；检查源码、数据、配置、卡数、LoRA目标及权重校验和，拒绝跨组恢复。
- 原完整组、无Event组、旧LoRA消融源码均未修改。备份见 `releases/p2_ltx_all_lora_components_20261109/MANIFEST.json`。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
# 只展示命令，不启动；另一组将no_feature换为no_whisper
"$PY" scripts/p2_ltx_all_lora_components_train.py commands --arm no_feature --stop-after 6000
# 用户安排资源后才执行；先最长样本两次真实更新、恢复原始状态，再正式训练
"$PY" scripts/p2_ltx_all_lora_components_train.py launch --arm no_feature --stop-after 6000 \
  --logs experiments/p2_ltx_all_lora_components_20261109/no_feature/nodes
```

启动前要求指定卡位空闲，禁止终止其他任务；STOP和已有run均被保护。恢复必须提供本组完整checkpoint。每组训练到指定6000保存退出，不自动接下一组。推理入口 `p2_ltx_all_lora_components.inference.load_model/sample`，不需要teacher或Event标签。

## 验收

`PREPARATION_ACCEPTED.json` 汇总证据。23项原生小模型测试包括全范围覆盖、禁用分支／优化器组／reader、AV及Event梯度覆盖、Feature对Connector的梯度、冻结基座、零增量前向和全组共享初始化、精确下一步模型／Adam恢复、跨组拒绝、CFG条件／mask／参考帧保留、CLI停止步数传递。完整22B结构仅在meta设备构建，未加载GPU。

每组 `preflight.json`、`META_COVERAGE.json`、`COLD_TRAIN.log`分别记录真实缓存、全模型覆盖和无GPU冷入口校验。冷入口应成功越过配置／源码／数据预检后，以“Real training requires GPU”明确停止；不能将其当作GPU运行验收。
