# UTMOSv2 六条音质筛选验证

使用用户指定的四条有杂音样本和两条干净对照，仅评分，不删除、不修改训练数据。

输入为 `listen_audio_cleanup_6_20261026/cases/<sample_id>/original.wav`，即官方独立 WAV 经既有规范化/裁切后的 listener 片段，不是 DeepFilterNet 或视听提取输出。逐文件 SHA256 已匹配上轮来源记录。完整 ID 与原始分数见 [results.json](results.json)，表格见 [results.csv](results.csv)。

## 配置与口径

- 官方 `fusion_stage3`、预训练 fold 0、数据域 sarulab。
- CPU float32，4 个 Torch 线程，batch=2；机器 GPU 正在执行其他任务，本试验不占 GPU。
- 完整音频输入，`remove_silent_section=False`，内部重采样为 16 kHz，不额外降噪或调音量。
- 使用官方模型内部随机取窗，seed=42/43/44/45/46 分别预测，报告五次均值与样本标准差。
- 标准差是同一模型的取窗波动，不是人工评分置信区间。未做五 fold 集成，不等同于论文的完整竞赛配置。
- 权重以官方严格 state_dict 加载通过；权重 SHA256、库版本与仓库 commit 见 [provenance.json](provenance.json)。本地 UTMOSv2 源码未修改；timm 安装到试验目录隔离依赖。

## 结果

|样本|人工反馈|预测 MOS 均值|取窗标准差|
|---|---|---:|---:|
|S1855|有杂音|1.840|0.273|
|S0300|有杂音|1.826|0.287|
|S0193|有杂音|2.037|0.216|
|S1252|有杂音|1.427|0.279|
|S0866|干净对照|2.774|0.366|
|S1921|干净对照|1.906|0.359|

干净组平均 2.340，有杂音组平均 1.782。这六条有一定排序趋势，但不能被单一阈值完全分开：有杂音 S0193（2.037）高于干净 S1921（1.906）。例如直接保留≥2分，会误排 S1921、仍保留 S0193。样本极小且由用户挑选，不能据此估算全量筛选准确率或确定通用阈值。

建议作为音质排序的辅助分数，暂不能单独承担串音淘汰。低分原因是否来自噪声、短发声、安静占比或域差异尚未分离验证。保留两条干净对照，避免把低分自动解释为 listener 音质差。

## 复现

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
PYTHONPATH="$PWD/experiments/listen_utmos6_20261026/deps" /private/software/conda/envs/qwenomni/bin/python scripts/score_utmos_six.py
```

源代码：[score_utmos_six.py](../../scripts/score_utmos_six.py)。本轮模型加载与30次片段评分共 90.6 秒（包含本次模型初始化下载阶段）。
