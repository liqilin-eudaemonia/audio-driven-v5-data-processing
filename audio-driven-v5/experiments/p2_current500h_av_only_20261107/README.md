# 最终500h上的原始Qwen条件＋仅AV loss

**用户已确认不加Whisper。实验已准备并通过CPU验收，尚未启动任何GPU训练，也没有设置自动接续队列。** 当前三机联合训练继续运行。先前代码已完整备份：[备份说明](../../releases/p2_before_current500h_av_only_20261107T045322Z/README.md)／[1667文件SHA验收](../../releases/p2_before_current500h_av_only_20261107T045322Z/VERIFIED.json)。

使用最终train＋dev选集 **284,218段／500.001259h**：train281,060、dev3,158，保留来源split，排除test。复用当前完整caption对应的原始未SFT Qwen2.5-Omni-3B guided H、Gemma指导文本和混合AV VAE缓存；H仍来自配对speaker audio＋listener首帧＋当前标注描述。listener净化音轨是生成目标，不是Qwen观测输入。不重新打标、解码或缓存。

损失为 **L = L_video_AV + L_audio_AV**。只更新Connector和LTX文本LoRA；没有Whisper分支，没有Feature decoder，不读取Event标签或listener teacher，也不在LTX hidden上计算Event。辅助损失权重在配置中显式为0。沿用旧AVOnlyModel初始化和前向；历史Event类别结构及冻结头仅为保留初始化顺序和状态兼容，训练不调用它，优化器不包含冻结参数。

从原始预训练LTX和相同seed20261023开始，不从6026/6000步联合模型或旧AV-only训练权重接续。后续resume必须是本实验自己的完整模型／Adam／rank RNG检查点，并严格匹配数据、源码、配置和world。已训练联合权重不能用于“干净”AV-only对照。

| 准备配置 | global batch | 完整一轮步数 | 尾部填充 | 机器配置（未启动） |
|---|---:|---:|---:|---|
| [32卡主配置](config.json) | 32 | 8882 | 6 | node12＋node8＋node4＋node11，各8卡 |
| [24卡备选](../p2_current500h_av_only24_20261107/config.json) | 24 | 11843 | 14 | node12＋node8＋node4，各8卡 |

32卡配置复用本轮完整模型最初的duration512调度，**全部sample index／noise ordinal与该初始32卡调度相同**。24卡重新按duration384组批；真实样本与各自noise ordinal一一保持，批次组成和尾部填充有变化，不能宣称24/32卡优化轨迹完全一样。两配置都每500步完整保存，最终不足500的末步另存；all全重计算、CPU预取2、512MiB FP64流式归约、GPU Adam、原学习率／warmup50／clip1保持。机器只是准备配置，正式启动前需按用户安排检查资源。

准入由最终冻结选集及完整sample list控制，不按历史`training_admitted`字段过滤dev。配置中旧data_config只保留原资产／几何定义；实际训练成员、缓存和caption均来自当前TRAINING_CACHE_INDEX及三个混合Reader。数据读取显式不打开Whisper、teacher或标签Reader。

## 验收

[完整验收](ACCEPTANCE.json) · [准备记录](PREPARED.json) · [21项CPU测试](tests.xml) · [缓存回读及完整数据准入复用](preflight.json) · [冻结源码身份](SOURCE_IDENTITY.json) · [启动命令预览](COMMANDS.json)。

- 21项CPU测试通过：微型原生LTX连续两次AV更新，Event头调用时强制报错仍通过；Connector／文本LoRA梯度正常，冻结参数保持；与旧AV-only初始化逐值一致；完整模型／Adam／RNG恢复后的下一次更新逐值一致；新进程冷启动依赖检查及错误resume拒绝通过。
- 全量样本成员、顺序、source split、duration schedule及sample/noise配对核验通过。复用已验收的相同全量缓存准入记录；32条真实缓存回读覆盖train/dev、2～11.95秒、旧缓存复用、转正后部分复用及新增样本。Qwen与Gemma共用最新完整caption提示文本；每次读payload仍执行原SHA／身份／形状验证。
- 当前正在训练的270份源码逐个核对不变；新功能位于独立p2_current_av_only包和CLI，未修改运行中的训练代码、配置或进程。
- **尚未做正式24/32卡GPU训练、显存与速度实测。** 首次正式启动前训练入口会做最长样本前后向检查；该检查不更新优化器，稳定步速和带Adam状态的峰值仍需在实际更新后验收。未将微型CPU测试当作完整GPU验收。

## 使用入口

项目根目录下，使用原虚拟环境Python：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_current500h_av_only.py commands
# 用户分配机器后，才执行下列显式启动命令；本轮未执行：
$PY scripts/p2_current500h_av_only.py launch --logs experiments/p2_current500h_av_only_20261107/launch
```

24卡使用`--output experiments/p2_current500h_av_only24_20261107`，其preflight复用相同32条缓存回读、完整缓存准入及数据身份，独立验证24卡调度。续训另加`--resume <本实验目录内完整step_*.pt>`。STOP会在当前更新后完整保存退出；TensorBoard写入各实验自己的tensorboard/目录，不替换当前训练的6006服务。

## 再次全量复核（2026-11-07）

[全量复核报告](RECHECK_FULL.json)核对全部284218条最终选集对应的caption原文件SHA与正文／来源／音频配方，以及每条Qwen guided H和Gemma指导文本的完成契约与caption哈希；全部一致。汇总为train281060＋dev3158，500.001259126h；24/32卡调度和源码身份核验通过。[21项回归再次通过](recheck_tests.xml)。本次没有重新读取全部大张量计算SHA；张量仍沿用既有完整验收与运行时逐payload校验。未进行正式多卡GPU训练。
