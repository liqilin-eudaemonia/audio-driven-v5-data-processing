**最新状态（2026-11-02）：285,638条已全部完成，失败0，独立全量验收通过。** 串行最终审计改为32个CPU进程执行同样检查，耗时25.14秒；结果见 `production/COMPLETE.json` 和 `production/parallel_audit_result.json`。四机GPU已接续新规划SFT guided H提取。下文保留执行与优化记录。

# 当前500h的 Feature teacher 补充缓存（四机32卡）

2026-11-02：正式提取运行中。范围为当前500h的 **285,638条train／500.0002847h**，沿用原标注自然语言 guided 条件和原模型。只补 AV＋Feature 的真实 listener 特征监督目标，不启动训练。目录名 `node8` 是最初单机安排，实际已扩展为四机。

## 数据、模型与语义

来源是 [原500h缓存](../p2_j_train500h_cache_20261101/)。读取冻结样本 source，核对画质测量 `crop.new` 及 `quality-bounded-crop-v1-20261030` 最终裁框，不读取审核网页视频。原清单 SHA256：`a247724de3a822c7a0ba7b2e2698c32a1f1bf5ebe5301b1ffa44c18af07ed82d`。

视觉路径：原TAR视频 → 最终listener裁框 → 512×512 → 按既有25fps训练时间网格选择8fps teacher帧 → teacher尺寸224×224。先裁后缩小，沿用原缺帧样本精确PTS特例。优化必须保持所选帧、像素和PTS一致。

音频路径：该listener的原归一化音轨与对应源区间 → 冻结的官方词／句／VAD并集净化配方（100ms边缘、10ms淡变）→ 保留完整时长和静音间隔 → 单声道16kHz。净化前后的身份核验与原500h记录一致；按原round网格处理至多一个多余端点。speaker音频只用于原训练条件，不作为teacher目标。

teacher是原始冻结 **Qwen2.5-Omni-3B** 的视觉塔和音频塔，权重 `/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B`，BF16、SDPA；不加载新规划LoRA。输出维度2048，视觉时间组4Hz（8fps两帧一组）、8×8空间网格，音频聚合8Hz。静音仍是有效监督，mask只排除无完整时间支持的尾部／padding。

## 存储与读取

- 新teacher：`production/cache/<sample_id>/payload.pt`、`complete.json`。
- 原 `input / input_plan / text_guided / vae` 全部复用，不重写；不新增WAV、MP4。
- `production/protocol.json` 固定来源清单、成员、模型、参考源码及媒体语义。
- `production/done/` 是每128条动态任务完成凭据，绑定逐样本身份、哈希、字节数。
- **当前四机进度：`production/cluster_status.json`**；历史 `cluster_history.jsonl`。旧根目录 `status.json` 属于已退役单机控制器，不再代表当前状态。
- 每机日志／配置／资源在 `production/nodes/<node>/`；分阶段计时在 `production/timings/`。
- 全部样本、来源身份和写入凭据验收完成后，监控进程自动生成 `production/COMPLETE.json`。运行中不能将该目录当作全部完成的缓存。

```python
from p2_j_teacher.core import Reader
reader = Reader('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_teacher500h_node8_20261102/production')
bundle = reader.training_sample(sample_id, variant='guided', device='cpu')
# condition/text/AV读取原500h缓存；teacher新增video/audio/video_valid/audio_valid。
```

原500h `p2_clip_events` 训练backend仍限制AV／AV＋Event。本次提供可组合读取接口和teacher缓存，**不代表原训练入口已直接支持AV＋Feature**；后续接入Feature损失时需要连接该接口。

## 四机配置与优化

| 节点 | IP | GPU | 全局worker编号 |
|---|---|---|---|
| node8 | 172.29.104.233 | 0–7 | 0–15 |
| node17 | 172.29.15.91 | 0–7 | 100–115 |
| node12 | 172.29.252.37 | 0–7 | 200–215 |
| node4 | 172.29.225.51 | 0–7 | 300–315 |

当前每卡2个模型进程、每进程8个CPU准备进程（每机128个）、预取32、调度group8、两个异步写入线程。group8仅用于调度和写入组织；视觉／音频前向均独立逐片段执行。共享文件锁动态领取128条任务，空闲worker继续领取后续任务，避免静态分片长尾。

GPU低负载主要来自视频解码供数：此前输入准备约90%时间在帧读取。已实施：只裁剪选中的帧；缩短seek回退且保留原读取器严格回退；并行准备和GPU执行重叠；完成顺序预取，已准备好的输入不被队首慢视频阻塞。CPU进程单线程解码，避免过多FFmpeg线程争抢160核配额。2／3／4模型进程配置均做过运行观察，更多模型进程不会自动增加吞吐。

参考语义代码 `src/p2_j_teacher/` 已冻结。执行层优化位于 `scripts/p2_j_teacher_cluster.py` 和 `src/p2_j_teacher_accel.py`。`fast_decoder.json` 绑定加速解码文件及验证报告SHA256；每条新payload的 `audit.execution` 保存所用执行版本。旧缓存保留、恢复时严格校验后复用。

可借鉴之前的解码、预取、跨卡动态调度经验；本任务不需要预计算T5，既有Gemma／Qwen条件已经缓存。GPU温度不是验收指标，需看有效产出、CPU供数和GPU利用率窗口均值。

## 验证与未采用的候选

- CPU基础检查4项通过：真实H264像素／PTS、近尾与选帧tie、静音mask、动态领取不重复；另新增完成顺序调度3项通过，覆盖慢队首绕过、已有缓存保留、异常不能吞掉、成员不丢失及有界提交。
- 正式基准是 **`pilot_v3/`**：32条真实样本包含此前人工复核样本、唯一已批准缺帧样本及固定随机补充。32条媒体像素／PTS／净化音频／processor输入一致，逐片段视觉／音频teacher与参照逐值一致；原guided缓存与teacher组合读取通过。8卡均完成group8／16／32测试。
- `decode_execution_verified.json`：32条使用新seek解码，与已验收基准的全部RGB哈希和PTS一致。
- `pilot/`、`pilot_v2/` 是被否决的原生拼批候选。视觉拼批最大相对L2约3.39%，音频拼批约4.47%，因此正式不使用这些数值有差异的方案。
- GPU NVDEC试验记录在 `nvdec_probe.log`：前两条像素／PTS一致，但后续源视频发生解码错误；**未部署正式任务**。
- 每次写入原子落盘、SHA256及全部特征／mask逐值读回校验；最后再独立检查完整成员及来源绑定。`failed/` 非空时禁止宣布完成。

## 状态与恢复

状态命令（工作目录为项目根目录）：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
WORK=experiments/p2_j_teacher500h_node8_20261102/production
$PY scripts/p2_j_teacher.py status --work "$WORK"
```

每机实际启动命令和所属PID在 `production/nodes/<node>/launch.json`。恢复前检查该节点本任务控制器，避免重复启动；只停止本任务所属进程。以下为node17配置示例，其他节点仅更改node和offset：

```bash
$PY scripts/p2_j_teacher_cluster.py node --work "$WORK" --node node17 --offset 100 \
  --per-gpu 2 --producers 8 --batch 8 --prefetch 32 --completion-order \
  --fast-decoder experiments/p2_j_teacher500h_node8_20261102/fast_decoder.json
```

已有全局watch进程记录在 `production/cluster_watch.json`，不要重复启动。worker异常可由节点控制器重启两次，失败任务会单独隔离并在状态中明确报告；失败不会被当作成功或静默丢弃。原完整payload不重新提取。所有结果成功后自动全量验收，不自动启动训练。

配置快照：[recommended_execution.json](recommended_execution.json)；代码备份：[source_snapshot_20261102/files.json](source_snapshot_20261102/files.json)；正式结果组合读取抽查：[production_readback.json](production_readback.json)。

## 历史缓存加速经验的对应关系

本轮参考的是已有缓存提取方案，而不是把打标推理的batch参数直接搬过来：

- [J新500h四机缓存](../../related/P2_J新数据缓存与四机32卡提取_20261101.md)：已复用每卡2模型进程、每进程8个独立解码进程、原ROI裁剪实现、异步写盘、共享动态任务及来源验证。本轮额外使用短seek和完成顺序预取。
- [28卡Feature压测](../../related/P2_特征抽取28卡配置压测.md)：双进程、CPU输入提前准备、expandable_segments已沿用；其约85.6% GPU利用率和约62%提速来自包含input、teacher、VAE等部件的完整流程，不能作为本次仅teacher补提取的直接速度基准。
- [全量特征接续与巡检](../../related/P2_全量特征抽取接续与资源巡检.md)：有界供数、吞吐优先、不要无限增并发、独立来源身份和完成验收均保留。本次没有恢复历史20分钟消息报告。

旧Gemma batch32已对应到原500h文本缓存，本轮无需重做；旧跨卡时长对齐用于DDP同步训练，本轮使用独立worker动态领取，没有每步全卡同步屏障。

最新窗口、资源测量和限制见 [optimization_summary.json](optimization_summary.json)。硬解码ROI候选也未通过：第一条真实样本RGB哈希不一致（`nvdec_roi_probe.log`），未用于正式缓存。正式采用经过32条像素/PTS一致性验收的CPU解码方案，不能用未经等价验证的路径换取利用率。
