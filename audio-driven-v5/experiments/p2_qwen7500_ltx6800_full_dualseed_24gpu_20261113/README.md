# Qwen7500＋LTX6800：完整Seamless与ResponseNet，三机双seed

**仅完成代码、清单和CPU核验，未启动GPU任务。** 根据用户最新指令，默认先完成全部推理及视频/音频导出，不自动评分、不做双seed汇总。

## 固定顺序与规模

| 顺序 | 数据集 | seed | 总条数 | 已核验复用生成 | 待生成 |
|---|---|---:|---:|---:|---:|
| 1 | Seamless完整test | 137 | 6624 | 256 | 6368 |
| 2 | Seamless完整test | 238 | 6624 | 256 | 6368 |
| 3 | ResponseNet完整配对集合 | 137 | 586 | 0 | 586 |
| 4 | ResponseNet完整配对集合 | 238 | 586 | 0 | 586 |

合计14,420条，复用512条，新增13,908条。Seamless的256条曾用于选权重，完整test论文报告需披露这部分调参重叠。ResponseNet仍为135个有配对片段的官方来源、586条，不是随机200；其中72条原有本地Whisper词时间及paper_comparable=false标记保持。

## 固定权重与原生参数

- Qwen：`../p2_planning_sft_continue2epochs20_20261113/training/checkpoint-7500`；复用256实测的adapter身份及504张量加载校验。
- LTX：`../p2_full_resume5500_to7000_after_ablation_20261112/train/step_006800.pt`，SHA256 `59de7ed52585228a37d66206cbcf512fc2b11cb527113e36c41aeac369fe7e45`；复用原完整checkpoint的已验收SHA/stat、mmap加载、4809模型张量与1632处全LoRA严格加载，不重导出几十GB权重。
- seed137、238；Euler30步、video CFG3／audio CFG7；512×512；Qwen7500自己的规划、hidden、Gemma。训练使用base Qwen hidden的事实不变，推理沿用这次256实验已经验证的7500条件。
- 规划greedy512、原生重试1024；仍截断时保留模型真实输出和标记，不伪造EOS、不删困难样本。
- Seamless全部25fps。ResponseNet维持各样本23.98／24／25／29.97／30fps及原帧数，按既有PIL bicubic640→512导出；检测padding64只作用于评分人脸检测。Seamless padding0。

## 缓存、加速与恢复

7210条speaker Whisper、输入首帧和source-only审核记录均已逐条核验复用。Seamless已有256份Qwen7500条件，需补6368份；ResponseNet需补586份。每个数据集的条件只提一次，两seed共享只读条件，输出及评分目录按seed完全隔离。

24个独立缓存分片，沿用planner batch16、Gemma batch8。缓存提取直接复用原listener首帧，并读取原speaker波形，核对与Whisper缓存的波形SHA一致，避免重复解码listener视频。长音轨SHA按文件stat在单worker内缓存，改变文件会失效。

推理阶段最多22路常驻原生生成＋2路常驻VAE解码；完成生成后空闲卡可以接更多解码。通过原生逐样本flock跨机领取任务、提交完成收据、跳过同身份已完成输出。完成索引只验证新提交，减少反复全量读取历史完成收据。失败最多重试3次，保留日志和断点；任一轮执行失败不会自动跳到下一数据集。

node12 `172.29.252.37:22`、node8 `172.29.104.237:22`、node11 `172.29.189.119:22`，各GPU0–7。**启动前要求24卡都已释放**；未释放时写RESOURCE_CHECK.json并退出，不停止其他任务、不静默降成两机。运行时按显存余量准入，cache/render/decode分别限制PyTorch80/96/38GiB，每卡最多本任务一worker。UUID和实际IP启动时核验，node4不参与。

## 命令

在仓库根目录 `/private/liqi_workspace/project_code/audio-driven-v5` 执行。以下启动命令目前没有执行。

```bash
# 只检查冻结代码、清单、配置
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/run.py check

# 只查询三机资源，不启动GPU计算
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/run.py resources

# 后台启动或从断点恢复四轮推理/导出，不评分、不汇总
nohup /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/run.py run >> experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/controller.log 2>&1 &
```

controller.lock防重复启动。创建本目录STOP，或对CONTROLLER.json中的已核验本任务PID发送SIGTERM，可停止本队列；控制器只关闭自己启动的远端worker。恢复前按用户意图移走STOP，重新运行上述命令，已提交输出继续保留。每轮INFERENCE_COMPLETE.json表示该轮生成与无损listener音轨导出都完成；本目录INFERENCE_COMPLETE.json表示四轮均完成，**不表示评分完成**。

## 输出位置

- `seamless/seed137/`、`seamless/seed238/`
- `responsenet/seed137/`、`responsenet/seed238/`
- 每条视频：`<dataset>/seed<seed>/generated/qwen7500_ltx6800/<dataset>/<pair_id>/generated.mp4`
- 同目录 `listener.wav` 是模型生成的无损listener音轨；metadata.json记录模型、规划SHA、seed和几何。
- `status.json`、各轮status.json、logs/、processes/、devices/、model_acceptance/记录进度和实际执行证据，启动前这些运行文件不存在。
- 每数据集`conditioning/`存7500条件，两seed共享；INFERENCE_COMPLETE和SCORING_COMPLETE分别标记生成与评分，不混用。

## 后续单独评分入口

完整四轮推理导出结束后，才允许显式运行：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/run.py score
```

此入口仍需24卡已释放；按四轮顺序执行既有视频/音频评分及各自全量合并。Seamless每seed可复用旧256的10个完整评分块，再补199块；ResponseNet每seed19块。FID/FVD按各轮全量特征计算，禁止平均分片分数。保留UTMOS、音频回应和表情/运动指标的原生N/A与实际覆盖。该入口不自动计算两个seed均值。

## 核验

`PREPARED.json`固定配置、清单、执行依赖与原已验收256代码指纹；`PLAN.json`记录规模和顺序；`SOURCE_WAVEFORM_ACCEPTANCE.json`验证两数据集长样本的speaker波形与原Whisper缓存逐SHA相同。`test_workflow.py`覆盖真实6624/586范围和帧率、跨seed误复用拒绝、source改变拒绝、22＋2调度与低显存准入、完整推理顺序/恢复跳过、失败不越过当前数据集、latent不误当完成视频。

GPU实机验证尚未执行，本次没有启动新推理任务。模型/采样/解码和指标来自既有已运行入口，新增加的是完整集合准备与调度适配。
