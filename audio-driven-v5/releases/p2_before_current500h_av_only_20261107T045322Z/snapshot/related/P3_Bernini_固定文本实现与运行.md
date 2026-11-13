# P3 / Bernini 固定文本实现与运行

更新：2026-10-21。对应[实验计划](P3_Bernini_固定文本倾听实验计划.md)。本文件记录实际实现、运行入口和验证状态；工程检查点不代表已训练出有效的倾听模型。

## 当前实现

代码独立放在 `audio-driven-v5/src/p3_listen/`，入口为 `scripts/p3_listen.py`，配置为 `configs/p3_listen.json`。复用 `../audio-driven-v4-p3-autoselect-20261013/src/` 中的在线 Bernini 模型、原生 LTX 条件拼接与采样实现，不修改历史工程。

| 模块 | 职责 |
|---|---|
| `dataset.py` | 按 session/参与者连通组划分，固定划分后切窗；自动事件仅进入独立 review 清单。 |
| `subturns.py` | 将 V4 长区间连续、不重叠地切成不超过 10 秒的子话轮，并统一原始参与者音轨响度。 |
| `media.py` | 从原始 MP4、归一化后的原始独立 WAV 导出训练素材；验证时长、帧数和哈希。 |
| `prompts.py` | Omni/Gemma 固定文本，训练和推理共用；不序列化 GT 回复或行为时间表。 |
| `features.py` | 原生输入/目标特征、Gemma 正/负文本和音视频 VAE 缓存；推理 layout 与 GT 特征分文件保存。 |
| `training.py` | 单卡 P→C0→A，梯度累积、预算、阶段迁移、严格恢复、冻结参数检查和开发集数值验证。 |
| `generation.py` | Pred25→回填→hidden→LTX；支持 text、oracle、prefix、audio-swap 诊断及完整 AV 解码。 |
| `evaluation.py` | 固定开发集对照任务清单、盲评表与方法映射。 |

主训练入口提供单卡实现，effective batch=8；另有经过实测的 8 卡 DDP 工程验证入口与 8 卡并行生成检查。没有启用旧多机自动调度、Whisper 分支、GT 文本计划或旧适配权重。正式阶段需按开发集结果选点后显式启动下一阶段，不自动开启 A。

## 数据修订：子话轮与音量

用户新增要求已覆盖初始方案中的数据处理设定：

- V4 的 457 个父区间切为 **872 个子话轮**，其中 208 个父区间被进一步分割。最长 **10.0 秒**，父区间内连续、不重叠；总时长仍为 **6574.949 秒**。
- 优先选择 speaker 停顿/词间隙并避开 listener 短反馈；33 个切点无合适词间隙，使用 10 秒硬切；1 个切点截断 listener 活动，均在审阅记录中明确标记。
- 对 100 条原始参与者完整音轨测量 EBU R128 响度，使用固定增益和峰值限制，再复测至目标。各条音轨最终为 **−18.29～−18.02 LUFS**，最大真峰值 **−0.97 dBTP**。同一音轨在 speaker/listener 角色以及各子话轮中使用同一处理结果；不按静音窗口单独归一化。
- 限幅使用 192 kHz 过采样，补偿本机 FFmpeg 4.2 的 959 样本前视延迟，随后回到 48 kHz。全部音轨长度核对通过；对原始音量低、中、高三条音轨的波形相关检查均测得 0 样本偏移。
- V5 预览视频从原始 MP4 重新裁剪，独立音轨来自归一化原始 WAV。完整会话上下文仍保留原始音量，页面明确标注。

当前数据：`data_previews/seamless_subturns_v5/manifest.json`。边界/响度/同步报告：`data_previews/seamless_subturns_v5/audio_boundary_validation.json`；全部切点：`subturn_audit.json`。

训练使用 `experiments/p3_fixed_text_v2_subturns/`。沿用最先固定的 10/2/2 连通组，不因新的子话轮数量改变划分。训练/开发/测试窗口分别为 **1160 / 224 / 228**；37 个不足 4 秒的片段保留为短片鲁棒性清单。训练窗口仍遵守初始最大 201 帧、25 FPS（8.04 秒）及 8k+1 帧结构，可在同一子话轮内重叠；这与供审阅的连续、不重叠子话轮清单是两个层级。

`experiments/p3_fixed_text_v1/` 是新增数据要求前的数据划分和少量接口检查产物；其分组被沿用，旧音量缓存不进入当前实验。

## 环境与数据准备

以下命令从 `/private/liqi_workspace/project_code` 执行。复用已有 Python 环境：`Omni-LTX-stage2/.venv/bin/python`，包含 PyTorch 2.9.1、Transformers 4.57.6 和本地 LTX 包。

```bash
# 原始话轮修订与响度处理：已执行，重跑会核对缓存。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/prepare_listen_subturns.py prepare --workers 6
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/prepare_listen_subturns.py render --workers 8
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/audit_listen_subturns.py

# 固定分组、训练窗口和 S0 选择。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py prepare

# S0：固定 16 个训练窗口和开发集窗口。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py export --s0 --split train
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py cache --s0 --split train --part all --gpu 0
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py export --s0 --split dev --limit 2
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py cache --s0 --split dev --limit 2 --part all --gpu 0
```

`--part native/text/vae` 可独立重试缓存阶段。去掉 `--s0` 和 `--limit` 可处理对应 split 的全量窗口。主训练前必须完成 train/dev 缓存与质量核查；不要在调参阶段处理或评估 test 输出。当前 test 仅有离线清单。

Gemma 固定文本缓存全局共用。Omni 缓存的是冻结的输入 embedding，不是可训练 Thinker 的最终 hidden。目标视觉特征使用 8 FPS；目标音频始终包含整段静音。推理 layout 仅由输入音频、首帧和窗口几何决定；缓存目标特征后还要核对其形状与独立构造的 layout 一致。

## 训练与恢复

```bash
# P 两步工程验证；先保存一步，再恢复完成第二步。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py train --stage P --s0 --steps 2 --stop-after 1 --gpu 1 --output audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_P_checked
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py train --stage P --s0 --steps 2 --gpu 1 --output audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_P_checked --resume audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_P_checked/step_000001.pt

# C0 从本实验 P 检查点初始化，优化器重新建立。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py train --stage C0 --s0 --steps 2 --gpu 1 --output audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_C0_checked --checkpoint audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_P_checked/step_000002.pt
```

A 使用相同入口，指定 `--stage A` 并提供 C0 检查点。`--s0 --oracle-fit --stage C0 --steps 200` 支持 GT 特征可见的小样本拟合诊断，输出必须另设目录；此类检查点不能用于主实验初始化。

正式训练去掉 `--s0`，从原始预训练权重重新开始 P；代码拒绝将 S0 检查点作为正式 C0/A 初始化。P/C0/A 上限分别为 20/10/5 epochs 或 2000/1000/500 次更新，以先达到者为准。检查点绑定配置、数据清单、固定文本、模型来源和实现代码；恢复时校验哈希并恢复优化器/RNG。

训练目录包括 `run.json`、`train.jsonl`、`step_*.pt`、对应哈希完成标记和 `status.json`。创建该目录下的 `STOP` 文件可在下一次更新前停止并保存。正式训练的数值验证只访问 dev；数值报告不能代替 Pred25 生成质量评估。

## 推理与对照

```bash
# 将 SAMPLE_ID 和 CHECKPOINT 替换为当前 windows.json 中的开发集 ID 和 C0/A 检查点路径。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py generate --split dev --sample-id SAMPLE_ID --checkpoint CHECKPOINT --mode pred --seed 137 --gpu 0
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/p3_listen.py matrix --split dev --checkpoint CHECKPOINT
```

- `--mode text`：原始 LTX、首帧、固定 Gemma 文本，无规划器。
- `--mode pred`：全 MASK Pred25，回填后读取 hidden，再连接到 LTX。
- `--mode oracle`：同检查点使用真实目标特征，仅作 GT 辅助诊断。
- `--mode prefix`：仅编码输入前缀；不传目标 hidden。
- `--mode audio-swap --swap-audio PATH`：替换为等长 speaker 音频，重新提取输入、规划并生成。音频必须预先裁成完全一致的采样长度。

推理直接编码首帧，不读取 GT 后续帧或目标 VAE。输出包含原始 latent、生成 MP4、波形及来源/几何报告。默认原生 LTX Euler30、video CFG=3、audio CFG=7，不加 STG/rescale/upscaler。

`matrix` 输出最多 156 个开发集任务、盲评 CSV 和独立映射文件。音频替换任务指定候选 donor 与精确目标时长；执行前需准备相应等长音频。生成任务需显式执行，盲评需人工填写，不会把自动 VAD 或单一 GT 时间对齐当作适宜性结论。

测试集生成还要求 `test_release.json` 记录锁定的 `config_sha256`（配置对象的 canonical JSON SHA256）及 `checkpoint_sha256`。这是实验隔离检查，不是额外审批流程。

## 验证

```bash
PYTHONPATH=audio-driven-v5/src:audio-driven-v4-p3-autoselect-20261013/src Omni-LTX-stage2/.venv/bin/python -m unittest discover -s audio-driven-v5/tests -v
```

11 项测试通过，覆盖跨 session 共享人物合组、划分可复现、窗口完整性、停顿切分与硬切回退、检查点阶段迁移和错误状态拒绝，以及删除 GT 特征文件后主推理输入仍可加载。

已完成真实 GPU 工程验证：

- 当前新版数据的 16 个训练、2 个开发窗口完成原生 AV 特征和 VAE 缓存；固定 Gemma 正/负文本编码完成。
- `runs/s0_P_checked/`：P 第一步保存后，严格恢复优化器/RNG 完成第二步，冻结参数未变化。
- `runs/s0_C0_checked/`：由 P 初始化完成两步；峰值显存约 48.65 GB。
- `runs/s0_A_checked/`：由 C0 初始化完成两步；第一步为暖启动，第二步联合规划器。第二步 Omni LoRA 梯度范数约 0.00357，DiT LoRA 约 0.00246，均确认梯度路径有效；峰值显存约 50.37 GB。P 的前两步受特征解码器多层零初始化影响，Omni LoRA 梯度为零，不据此误判永久断开。
- `validation_8gpu/ddp/completed.json`：8 张 H200 上真实 DDP A 阶段两次联合更新，每步 8 个 rank 的可训练参数 SHA256 完全一致，有限损失/梯度、冻结参数未变化；第二次全局更新约 3.94 秒，单卡最大分配显存约 50.97 GB。这是两步工程测量，不是长期吞吐基准。
- `validation_8gpu/generation/completed.json`：8 卡同时运行 8 个完整生成任务，8/8 成功，总墙钟约 106.96 秒。覆盖原始文本基线、C0/A Pred25、C0/A Oracle、prefix 和音频替换；包含 185 帧/7.4 秒及 201 帧/8.04 秒两种窗口，512×512，48 kHz 音频。LTX 原生因果音频解码产生的 30 ms 尾部补齐已明确记录，不属于输入音轨错位。
- `data_previews/seamless_subturns_v5/media_validation.json`：5282 个媒体文件通过，50 个最长双人对照片段完成全解码。`browser_validation.json`：播放、独立音轨、上下文、筛选、审阅导出、Range 下载和移动端宽度均通过。

预览导出检查发现 FFmpeg 4.2 的 `-frames:v` 会使同次封装的音频提前结束，已通过第二遍独立封装完整音轨修复（视频流直接复制，不再次编码）。`scripts/repair_listen_preview_audio.py` 已接入导出入口，发布程序要求该步骤完成；独立 M4A 和训练 WAV 不受该问题影响。

8 卡验证复现：

```bash
Omni-LTX-stage2/.venv/bin/torchrun --standalone --nproc_per_node=8 audio-driven-v5/scripts/check_listen_ddp.py --checkpoint audio-driven-v5/experiments/p3_fixed_text_v2_subturns/runs/s0_A_checked/step_000002.pt --output audio-driven-v5/experiments/p3_fixed_text_v2_subturns/validation_8gpu/ddp --steps 2
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/check_listen_8gpu_generation.py
```

运行新的生成验证时需另设输出或保留已完成结果后调整脚本中的输出目录，避免覆盖已有样本。用户授权清理的 GPU 服务/实验调度进程清单在 `resource_cleanup.json`；只清理计算工作负载，保留系统与远程连接服务。8 卡验证结束后释放 GPU，不用无意义任务维持占用。

当前尚未完成正式 P/C0/A 训练、200 步 Oracle 拟合、开发集完整对照和人工盲评；不能据工程链路通过判断固定文本方案已有效。
