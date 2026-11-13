# LoRA6000＋SFT2752：Seamless完整test6624，node11 GPU4

**Gemma显存限额故障已修复并断点恢复（2026-11-12）：** 先前Gemma batch16在48GiB单进程配额内堆叠隐藏状态失败，当时整卡仍有约47GiB空闲。保留全部6624条Whisper／Qwen缓存、196条复用生成及4条新增latent；改为Gemma batch8、缓存进程64GiB配额，编码器、精度、采样与指标不改。后续若再遇真实Gemma OOM，会自动进一步减小batch再试；失败状态现在明确写为failed。旧源码、协议和异常保存在revisions/，见[恢复记录](RECOVERY_20261112.json)、[有效调度参数](RESOURCE_POLICY.json)、[实际运行配置](CACHE_RUNTIME.json)。Gemma BF16批次变化可能带来通常的舍入差异，不宣称与旧batch16逐位相同。

用户2026-11-12明确要求在node11物理GPU4启动。固定最新[SEAMLESS_CURRENT.json](../../evaluation/SEAMLESS_CURRENT.json)的完整6624条，采用upright-evaluation-20261112版本；仅此卡，GPU7的ResponseNet6000／6500队列及其他任务保持。控制器与子进程后台运行，以文件锁、逐样本提交和就绪队列断点恢复，不依赖当前聊天连接。

## 权重与数据

- 原始检查点：`../p2_ltx_all_attention_ffn_lora31_20261108/step_006000.pt`，SHA256 `3c2c78e8b1c81e3cf84efdc565b16a28565c0e923cacf8e902c3adcd6232eea6`。复用已逐张量核验的推理导出，attention／FFN共1632处LoRA。
- 规划器：Qwen2.5-Omni-3B的SFT2752，沿用原规划／H／Gemma／speaker Whisper实现；只输入speaker音频与listener首帧。
- 清单：`../seamless_upright_eval_fix_20261112/prepared_full6624/references.jsonl`，SHA256 `43fb951bcd7dcf36ab941a559a0eaf556ad4686a13dff2fca63b9c389f77432a`；6624条，身份互斥，不改划分。
- seed137、Euler30步、videoCFG3／audioCFG7，512×512／25fps；原生精度、VAE补帧／裁尾、净化音轨与评分公式保持。
- 与旧历史全集相比有33条转正来源更新。当前GLARE19050全量是旧来源版本；后续表格必须标明版本差异，不能声称33条的模型输入相同或静默混合排名。

## 复用与调度

旧全LoRA6000的固定200中，196条来源未改变，核验原检查点、seed、采样及源行身份后复用生成视频和listener WAV；4条转正来源不能复用其旧生成。本次需补齐6428条生成。

200条SFT／Gemma／Whisper缓存可复用：196条旧缓存和4条已按新来源重新提取的缓存（只复用条件，不复用其他生成器的输出）。其余输入优先复用已经完成的GLARE首帧／音频审计，33条涉及新来源的片段使用当前数据读取器；不会先导出整段GT视频或全量RGB。

CPU输入准备8线程，Whisper／Qwen／Gemma batch16；缓存与常驻生成进程重叠。使用追加的`ready_queue.jsonl`逐项发布完成条件，生成从队列领取，失败恢复跳过已提交产物。缓存完成后接原生常驻VAE解码和就绪32条视频／音频评分块，共207块。评分沿用batch64、GPU检脸／表情、CPU读取／像素预取、4个像素线程和GT特征缓存；全局FID／FVD合并完整特征，禁止平均分片分数。不额外存完整6624的RGB缓存。

每个进程固定`CUDA_VISIBLE_DEVICES=4`并校验GPU UUID；按实时可用显存启动后处理，生成期间最多1解码／1评分，生成结束后最多2解码／4评分。原生模型与采样不改；显存不足等待，不停止其他任务。

## 进度、结果与恢复

- [状态](status.json)、[固定协议](protocol.json)、[范围及复用验收](PREPARED.json)、[提交记录](LAUNCH.json)、[实际控制器](CONTROLLER.json)、[真实模型加载验收](MODEL_ACCEPTANCE.json)。
- `progress/cache/00.json`：Whisper／Qwen／Gemma当前进度；`progress/render/00.json`：新增生成进度；`progress/decode/`、`progress/metrics/`：后处理进度。
- 视频及无损listener音轨：`generated/ltx_all_lora_event_feature_whisper_6000_sft2752/seamless/<id>/generated.mp4`、同目录`listener.wav`。
- latent：`latents/ltx_all_lora_event_feature_whisper_6000_sft2752/seamless/<id>/`。复用的历史目录保留软链接，不能删除原200条缓存／生成目录。
- 评分原始行及特征：`metric_shards/<arm>/seamless/`；最终完整成绩：`RESULTS.json`、`metrics/<arm>/seamless/summary.json`及逐条JSONL。
- 最终完成以`COMPLETE.json`和`FINAL_AUDIT.json`为准。源缺帧或检测支持不足保留原生N/A及覆盖，不伪造或填0。
- 日志：`controller.log`、`processes/*.json`指向的`logs/*.log`；故障写`FAILURE.json`或`<stage>_failure.json`，逐worker最多重试3次，已完成产物保持。

入口：[p2_lora6000_seamless_full_gpu4_20261112.py](../../scripts/p2_lora6000_seamless_full_gpu4_20261112.py)。在node11项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u \
  scripts/p2_lora6000_seamless_full_gpu4_20261112.py controller
```

恢复前检查本目录CONTROLLER／processes中的PID是否仍存活；不要叠加第二份模型。如果控制器退出但已有子进程存活，先检查其提交进度再按本任务进程组清理；禁止按GPU或名称泛杀。用户设置本目录`STOP`时，控制器会停止本任务进程组；撤销STOP须得到恢复指令。
