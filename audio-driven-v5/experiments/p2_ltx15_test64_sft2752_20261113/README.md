# 15份LTX权重初筛：test64＋SFT2752

**四机扩容已启动，当前三机21卡：** node12/node8各0–7、node11为0–4。node4 SSH暂不可达，8卡待恢复后加入；交接已保留128条视频／129条latent。旧单卡控制器已退出，恢复使用cluster29入口。[运行与恢复](cluster29/README.md)。

用户授权暂停GPU4的GLARE10618完整测试，优先运行本轮。GLARE暂停时已生成2223/6624条、14个完整评分块，产物保留；暂停收据见 `GLARE_PAUSED.json`。本任务不使用GPU7，不自动恢复GLARE。

## 范围和固定条件

- 权重：`../p2_full_resume5500_to7000_after_ablation_20261112/train/step_005600.pt` 至 `step_007000.pt`，每100步，共15份；**同一次28卡续训**，不是历史同名6000/6500/7000。
- 数据：`../seamless_test_tuning64_20261113/prepared_64/`。同64条×15＝960条；这是用户指定的test调参子集，不是独立dev。
- Qwen：原已完成的SFT2752，固定其规划文本及hidden states；64条Qwen、Gemma、Whisper、首帧缓存均复用并核验来源。SFT2752是比较基准，并不代表已确认最优。
- 推理：seed137、Euler30、video CFG3／audio CFG7，512²／25fps，保留原生时间、随机数和模型计算。
- 四个按时长平衡的评分块／权重；按64条全部特征重新计算FID/FVD，不平均分片FID/FVD。GPU检脸及情绪、batch64、像素4线程，读帧预取和跨权重GT特征／无损RGB缓存。

## 提速和恢复

按完整保存标记核对15份step／模型结构／Adam步数／28RNG／训练源码与身份。CPU单路顺序核验完整SHA，与生成重叠；每份通过后才载入。直接mmap原始检查点，Adam不进入GPU；避免额外复制15份大模型文件。

一个常驻LTX生成进程严格切换4809份状态张量，复用22B预训练主干；旁路一个常驻VAE解码进程和一路评分，采样结束后最多两路解码、四路评分。显存门槛控制重叠，不修改采样质量。仅GPU4。已完成样本按原子收据复用，失败有界重试三次；不能把进程退出当作成功。

- 进度：`status.json`、`render_status.json`、`step_*/progress/`
- 检查点／来源：`CHECKPOINTS.json`、`PLAN.json`、`integrity/`、`model_acceptance/`
- 生成媒体：`step_<六位步数>/generated/continuation28_step<步数>_sft2752/seamless/<id>/generated.mp4`，同目录`listener.wav`
- 各权重评分：`step_*/RESULTS.json`及`metrics/`
- 汇总：`RESULTS.md`、`RESULTS.json`；全部完成后写`SCREENING.json`和根目录`COMPLETE.json`
- 日志与进程：`logs/`、`processes/`、`CONTROLLER.json`

暂停时创建根目录`STOP`，控制器停止自有子任务并保留完成输出。恢复前确认旧控制器及自有子进程已退出、GPU4仍获授权，再移除本目录STOP，执行：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx15_test64_sft2752_20261113/run.py controller
```

## 初筛规则

全部15份同64条评完后再做保守筛选。只有某权重相对另一权重同时满足LPIPS差至少0.02、FID和FVD差至少10%，且表情准确率、ΔResp和音频漏回应／误发声没有达到预设容差之外的优势，才列入“明显劣势候选”。具体容差在生成前写入`PLAN.json`。画质指标要求同64条完整覆盖、音频无执行错误；否则不自动淘汰。

这是工程初筛阈值，不是统计显著性或最终最佳证明。接近、指标互有优劣或支持不足的权重保留；不强制凑出top-k、不按训练loss筛选。原始权重永远保留。之后可扩样、补seed及交叉比较Qwen。
