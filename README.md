# audio-driven-v5 数据处理备份

**当前 P2 训练固定版（2026-10-26）：** [AV＋Event＋text-attention LoRA＋H200 优化＋TensorBoard：备份、验证及恢复说明](p2_training/av_event_lora_tensorboard_20261026/README.md)。保存实际使用的 P2/v4/LTX 源码、配置和依赖记录；备份副本 146 项 CPU 测试、两进程训练与监控检查通过。固定标签 `p2-av-event-text-lora-h200-tensorboard-20261026`。此前无 LoRA 标签不覆盖。

**P2 AV / AV＋Event 训练代码（2026-10-26）：** [独立源码快照、配置、验证与恢复说明](p2_training/av_event_20261026/README.md)。显式关闭 feature decoder、teacher 读取、feature loss 和对应优化器组；首轮比较事件监督是否有效。版本标签 `p2-av-event-no-feature-20261026`，简化版真实 GPU S0 尚未执行。

**RealTalk 切分边界 v2 已获确认（2026-10-25）：** [代码、规则、长视频结果与独立恢复](realtalk/boundary_v2/README.md)。完整611.64秒原视频生成137条／401.56秒候选，旧候选帧全部保留。原生裁剪、切点上下文和完整时间账本已审核；12项边界测试及恢复校验通过。全量切换、干净音轨认证与训练准入尚未执行。

**当前打标信息原文整合（2026-10-25）：** [独立代码、规则、50段输入结果与CPU重放](event_labeling/integration_verbatim/README.md)。保留模型原文，减少跨窗重复，自动选择已有时间，来源分歧单列；175条候选、682条输入观测完整保留。原有打标基线继续保存在event_labeling，当前整合入口以此为准。

**Listener 开放事件打标（2026-10-24，当前方案已获用户认可）：** [代码、prompt、复现与结果入口](event_labeling/README.md)。Qwen3-VL 视觉＋Qwen3-Omni 纯音频＋ASR 逐词时间，50 段的 1,089 条候选整合为 682 条开放事件。独立目录保存源码、原始响应、标签和来源追踪；支持无 GPU 逐字节重放。

**Seamless 四机无切镜全量已完成（2026-10-23）：** 64,087 对来源全部有结论，63,895 对结果（含 2,074 对源无效诊断）、192 对归一化异常隔离，剩余 0。训练主池音质候选 **2,002.627 小时**，起始反应储备另有 **333.317 小时**。四机 32 卡含恢复与导出共 **61 分钟**；未做人脸尺寸筛选或最终裁剪。[尺寸分布报告](data_corpora/seamless_listen_local_full/size_distribution_report.md)、[完成核验](experiments/p3_listen_local_full_20261023/completion_verification.json)、[代码与运行说明](scripts/README_listen_v8.md#seamless-无切镜全量完成2026-10-23)。最终清单索引及哈希已备份，完整大清单和媒体仍在源机器。下方为历史过程记录。

**Seamless 完整源视频回收全量版（2026-10-23）：** 用户已认可连续原片与前后 3 秒边界审核。新增 [全量入口](scripts/run_listen_local_full.py)、[处理规则与恢复说明](scripts/README_listen_v8.md#完整源视频全量运行2026-10-23)、[原项目总索引快照](SOURCE_PROJECT_README.md)。范围 64,087 对，3 机 24 卡最多 96 个模型进程；局部失败继续提取子片段，保持原音质基线，父子不重复计数。人脸和头肩参考框只做统计，不作为准入条件。35 项测试通过，正式入口复现已审核的 28 段/145.254 秒，并对未审核来源完成 6 段新切片质检。[验证证据](experiments/p3_listen_local_full_20261023/validation.json)、[冻结契约](data_corpora/seamless_listen_local_full/contract.json)。GPU 解码支持的样本已验证原缩放输出逐帧一致，不支持的格式回退 CPU。备份包含代码、小批元数据和验证，媒体、模型和全量动态结果留在原机器。以下为历史版本记录。

**RealTalk（2026-10-23）：** 新增独立 [代码与复核证据备份](realtalk/README.md)，规则 `realtalk-pairs-10s-max11-v1`；692 条视频、129,094 个候选 pair、110.7779 小时。包含原生框尺寸统计与恢复入口，尚未执行分辨率/音质筛选或全量视频导出。

2026-10-23 08:00 UTC 更新：修复节点目录暂时不完整造成的误停，运行预留显存改为 12 GiB，恢复三机 4×2。最近 5 分钟实测约 650 对/分钟；切分/音质/人脸统计规则和 GPU 模型批大小不变。新增仅针对已知调度故障、含资源和重试上限检查的运行监控。12 项调度测试、2 项恢复测试通过；历史停止与代码契约已归档。以下为此前版本的历史说明。

调度更新：新增 `run_listen_v8_audio_governor.py`，保留每卡最多 4 进程，在持续高负载时分级降档、稳定余量下恢复，并比较每进程 1/2 路准备的吞吐。只替换 coordinator，处理进程和科学契约不变。7 项调度测试通过；运行中的速率记录保留在原机器，备份切换证据见 `cluster/governor/`。

最新更新：2026-10-23 已按用户授权接入独立全量处理，移除视觉准入、保留人脸尺寸统计；入口 `scripts/run_listen_v8_audio_full.py`。音质规则保持 `v8-audio-severity-review-2`。三机每卡最多 4 进程，资源压力降档；最终裁剪暂不执行。新入口 7 项测试通过，历史 84 段分组/原因一致，20 段实际音质试跑成功。运行契约及试跑证据已归档，动态进度保留在原机器。以下“未替换全量入口”等描述为此前音质基线备份时的状态。

备份时间：2026-10-23。音质工作基线：`v8-audio-severity-review-2`。

本仓库保存当前 V8 完整语句切分、ASR 区间误拦恢复、边界分级、音质三路分流、人工反馈约束及相关调度/导出代码。源目录为 `/private/liqi_workspace/project_code/audio-driven-v5`；源代码逐字节保留，文件校验见 [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json)。

- [当前规则、阈值、代码导航](scripts/README_listen_v8.md)
- [最终音质路由](scripts/listen_v8_audio_final.py) 与 [必需任务/人工约束](scripts/listen_v8_audio_first.py)
- [切分策略](scripts/listen_policy_v8.py)、[规划](scripts/plan_listen_v8.py)、[恢复修复](scripts/listen_v8_recovery.py)
- [阶段决定](data_reviews/listener_v8_audio_policy_checkpoint_20261023.json) 与 [历史验收记录](related/P3_V8_QC4_第二批音质验收.md)
- [原始代码与证据快照](experiments/p3_listen_v8_audio_final_20261023/recorded_baseline/manifest.json)

明确串音和严重噪声排除；轻微底噪及短暂弱匹配按条件保留；证据不明确者待复核。人工正例只覆盖精确样本与区间，负例保留重叠区间约束。当前版本完成小批验证，尚未替换旧 QC4 全量入口；待复核不计入可用数据。本次备份不修改音频或裁剪，不启动全量任务。

仓库包含用于针对性测试的样本元数据、反馈和契约，不包含视频、音频、模型权重、全量中间产物或凭据。元数据中的绝对路径和预览 localhost 链接指向原机器，克隆后不会自动得到这些媒体。归档里的其他历史文档链接可能指向未纳入备份的大型实验产物。

在原有 Python 环境下，从本仓库根目录检查核心回归：

```bash
OPENBLAS_NUM_THREADS=1 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_listen_v8_audio_final.py -v
OPENBLAS_NUM_THREADS=1 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_listen_policy_v8.py -v
OPENBLAS_NUM_THREADS=1 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_listen_v8_recovery.py -v
```

完整运行另需原始 TAR/索引、归一化 WAV、QC3/QC4 和恢复审计缓存、AST/PANNs、FFmpeg 等；模型重算需匹配的 CUDA 环境。这里不是脱离数据与环境的独立可执行包。`prepare_listen_v8_audio_final.py` 会根据当前历史展示记录重新选样，精确复查旧小批应使用归档 selection/review_selection；不要把重跑准备脚本当成原样回放。
