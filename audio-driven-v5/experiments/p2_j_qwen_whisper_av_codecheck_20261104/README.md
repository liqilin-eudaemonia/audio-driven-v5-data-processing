**缓存提速v2追加验收：** 80项CPU回归通过；跨批次预取、编码／写盘重叠、数值检查门控及异常传播均已覆盖。真实样本与原缓存hidden最大误差0。[提速交接](../../related/P2_Whisper缓存流水线提速与接续_20261104.md) · [测试报告](prefetch_v2_tests.xml) · [真实样本对照](prefetch_v2_real_sample_regression.json)。当前已重新挂自动接续，GPU吞吐待运行后实测。

# Qwen H＋Whisper，仅 AV：代码验收

状态：代码已实现；**未启动正式训练，未提取全量 Whisper 缓存**。

完整交接：[P2_QwenH加Whisper双路条件_AV训练交接](../../related/P2_QwenH加Whisper双路条件_AV训练交接_20261104.md)。

- 配置：[实验配置](../../configs/p2_j_qwen_whisper_av_20261104.json)、[四机配置](../../configs/p2_j_qwen_whisper_av_32gpu_20261104.json)。
- 入口：[p2_j_whisper_av.py](../../scripts/p2_j_whisper_av.py)。
- CPU回归：**59项通过，0失败／0跳过**；含本实验22项及原仅AV／矩阵训练回归。[JUnit原始报告](cpu_tests.xml)。
- 两进程Gloo检查：两个完整AV更新后，所有已保存参数与串行两样本FP64累积更新逐值一致。
- 源码隔离：私有LTX复制111个Python文件；仅3个文件加入Whisper时间信息；复制后上游111个文件SHA256均未变化。[复制来源](../../external_sources/p2_whisper_av/UPSTREAM.json)。
- 真实样本：[real_sample.json](real_sample.json)。

真实样本为 `improvised_V00_S0822_I00000920_local_1ebda932114f`，来自确认的500h训练清单。冻结Whisper-large-v3在CPU实际提取并保存后读回：`[1,156,1280]`，20ms网格，0–3.115秒，全有限、无梯度。原Qwen H为 `[1,4,481,2048]`，Gemma标注文本及原AV latent可读。该单条Whisper缓存位于本目录的 `whisper_cache/`，不冒充全量缓存。

复验命令（无需GPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m pytest \
  tests/test_p2_whisper_av.py tests/test_p2_j_av_only.py tests/test_p2_matrix16.py -q
```

待完成：全量Whisper缓存、22B模型GPU前后向／显存检查、正式32卡NCCL检查、长训及ResponseNet200评测。当前没有宣称新增分支能沿用旧显存规格或已有质量收益。
