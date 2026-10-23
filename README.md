# audio-driven-v5 数据处理备份

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
