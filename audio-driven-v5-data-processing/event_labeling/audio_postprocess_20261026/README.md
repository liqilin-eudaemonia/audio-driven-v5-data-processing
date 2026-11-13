# Listener 音频后处理与训练读取备份（2026-10-26）

本版本保存已在固定50h训练子集上运行的代码：**不生成处理后WAV，保存静音配方、新音频latent与事件缓存；训练读取原视频latent＋新音频latent＋新事件标签。** 原音轨、视频和旧缓存保留。

30,415段全部完成并通过验收，保留23,958个音频事件，12,300段有发声标注；六类视觉事件不变，七类仍覆盖。新音频payload约1.26 GB，新增WAV和视频缓存复制均为0。此次备份没有启动训练或重新处理数据。

## 文件导航

| 内容 | 入口 |
| --- | --- |
| 规则、三机运行与训练读取 | [完整说明](snapshot/audio-driven-v5/related/P2_50h音频后处理三机运行说明.md) |
| 虚拟音轨与音频独立缓存设计 | [设计与契约](snapshot/audio-driven-v5/related/P2_虚拟静音音轨与音频缓存增量更新.md) |
| 核心包 | [listen_audio_postprocess](snapshot/audio-driven-v5/src/listen_audio_postprocess) |
| 单机命令 | [p2_audio_postprocess.py](snapshot/audio-driven-v5/scripts/p2_audio_postprocess.py) |
| 三机24卡、96分片与接续准备 | [集群入口](snapshot/audio-driven-v5/scripts/run_listener_audio_postprocess_cluster.py) |
| 训练读取适配 | [training_data.py](snapshot/audio-driven-v5/src/p2_listen/training_data.py)、[overlay.py](snapshot/audio-driven-v5/src/listen_audio_postprocess/overlay.py) |
| 固定100条语义回归 | [回归脚本](snapshot/audio-driven-v5/scripts/verify_listener_audio_bulk100.py) |
| 新训练配置 | [view/config.json](snapshot/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/view/config.json) |
| 30,415段完成与七类统计 | [COMPLETE.json](snapshot/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/COMPLETE.json) |
| 全部样本审计 | [audit.json](snapshot/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/audit.json) |
| 固定100条GPU对照 | [gpu100_acceptance.json](snapshot/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/gpu100_acceptance.json) |
| 存储及读取约定 | [storage_contract.json](snapshot/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/storage_contract.json) |
| 文件来源、大小、SHA256 | [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json) |
| 备份副本测试 | [restored_cpu.log](validation/restored_cpu.log)、[验收汇总](BACKUP_VALIDATION.json) |

## 处理规则

- 官方listener词／句／VAD并集决定保留声音范围；两侧0.1秒保护，内部边缘10毫秒淡变。并集外静音，缺失metadata报错，合法空并集为等长静音。
- 使用现有事件与保留区间对齐；唯一同词且边界接近时采用官方边界，否则取原区间交集。保留模型原文，不用官方转写替换不同词。
- ASR原文含两个及以上汉字的事件被移除并静音；单汉字保留。不扩展为所有非英语内容检测。
- 不连续支持区间保留内部边界，不把多个yeah／okay合成持续长发声。视觉事件不修改。

## 备份范围与依赖

`snapshot/`内保留三个同级目录：本轮P2/音频后处理/相关整合源码、实际使用的v4复用源码、LTX core/trainer/pipelines源码。保留现用配置、针对性测试、50h选集ID与统计、轻量处理契约及验收证据。依赖源码来自实际工作树，按文件哈希记录，而非仅依赖上游版本号。

未上传媒体、模型权重、训练checkpoint、latent载荷、全量官方metadata和完整标签、凭据。代码中原机器地址和密钥**路径**用于说明运行环境，私钥内容没有纳入备份。实际运行仍需要原数据与缓存；此目录不是独立的数据集镜像。

原机器数据入口：

```text
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_train50h_audio_postprocess_20261026/view/config.json
```

配置中的`data.audio_postprocess`选择新音频；`targets.json`中的WAV路径仍表示原始来源。直接播放原WAV不会得到静音后声音。若需要波形，调用`recipes.read_listener_audio`在内存中处理。

## 检查与恢复

完整性检查不需要GPU或模型包：

```bash
python verify_backup.py
```

用已安装依赖的Python检查备份副本；脚本隐藏CUDA并核实导入来自本快照，不安装环境，不启动训练：

```bash
P2_PYTHON=/path/to/python bash restore_cpu_check.sh
```

拷贝时保留整个本目录及三个同级snapshot目录。恢复到新位置可以运行CPU测试；真实数据处理需对应权重、媒体、缓存和路径。配置及运行脚本包含原机器路径与节点信息，换机器应建立新运行配置并重新验证数据／缓存身份。

已有缓存和checkpoint使用严格的源码、路径、模型、配方与数据身份。恢复旧运行时需恢复匹配的原路径和依赖，不能篡改完成标记或删除哈希检查来强行复用。此版本只支持AV＋Event、`feature_loss_enabled=false`；旧teacher音频特征不能与处理后音频混用。

## 验证的含义

原机器完成30,415段来源、缓存和标签审计，另101段实际训练读取通过；固定100段波形与试听WAV逐采样点一致，98段更新后的GPU缓存与实体WAV编码逐位一致，另2段复用未变化的原音频缓存。相关脚本和结果一起保存。

备份副本另行运行文件哈希、CPU规则与读取适配测试，结果见上方验收记录。没有在备份目录重新加载模型或执行训练；这些检查不证明生成效果。

固定版本入口：[GitHub `listener-audio-postprocess-train50h-20261026`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/listener-audio-postprocess-train50h-20261026/event_labeling/audio_postprocess_20261026)。本备份副本 **54项CPU测试通过**，418个snapshot文件哈希校验通过，生产运行的14个核心文件逐字节匹配。
