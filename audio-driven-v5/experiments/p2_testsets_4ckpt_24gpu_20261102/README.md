# 当前范围：四机32卡，只跑四个权重的ResponseNet（2026-11-02）

**已全部完成：2026-11-02 14:25:22北京时间。** 四权重共2,344条视频及音频评分、128分片与最终统计全部完成，无执行错误。UTMOS因参考语音过短为N/A。[汇总结果](RESULTS.json) · [完成标记](COMPLETE.json) · [最终推荐配置](../p2_metrics_acceleration_20261102/README.md#最终采用配置与完成状态)。以下时间估计和运行状态属于历史记录。

**当前评测加速：** 已启用GPU检脸／关键点／情绪、共享GT缓存与并行像素指标。172.29.189.119先使用空卡8进程；全部生成完成后自动加入四台推理机器的已释放GPU，每卡上限已从2增至4进程，batch仍为64。旧CPU评分归档，正式结果统一新后端重评；最终汇总仍在172.29.189.119。[实测与数值差异说明](../p2_metrics_acceleration_20261102/README.md)。

按用户最新要求加入node8，推理使用node17／node12／node4／node8共32卡。只完成ResponseNet全部586条×4权重＝2,344条，seamless暂停。逐权重顺序不变：AV＋Event 6,000→8,927，再仅AV 6,000→8,927；指标计算仍在172.29.189.119空卡。

586个ID重新按帧数均衡分到32卡，每卡18–19条，缓存及已完成采样／视频均复用。切换后确认首组已有564条采样、24条导出。新增rank24–31复用相同空提示缓存（原24份SHA256一致），评测相应调整为4×32＝128分片。原24卡配置及分片保存在scope_history/expand_32_*，目录名中的24gpu仅是历史名称。

按当前实测采样约0.224秒/帧，后续三组采样的理想计算量约46分钟，加解码、模型加载及组间切换，预计全部生成及导出还需约60–80分钟；首组约5–10分钟。指标会与后续推理并行，最终汇总时间另看评测机吞吐。实时状态见status.json和metrics_status.json。

以下保留原双数据集方案和执行记录，当前范围以上方及protocol.json为准。

---

# 三机24卡逐权重推理＋独立机器九项评测

2026-11-02，按用户新安排替代原四机队列。两组训练已完成8,927步。当前三机24卡的48条真实样本验证已通过：规划与文本缓存、AV生成、解码导出均成功，输出512×512、原FPS及GT帧数。全部7,210条共享条件缓存已完成，24个worker均通过；现已正式开始AV＋Event step6,000的三机24卡推理。

## 机器与执行顺序

推理只使用 node17（172.29.15.91）、node12（172.29.252.37）、node4（172.29.225.51），每机8卡。node8不参与GPU推理。SSH使用各机22端口、已验证私钥及共享路径。

1. 一次提取所有样本的未微调Qwen规划、Qwen hidden、Gemma正／负文本缓存，四组共享。
2. AV＋Event step6,000：24卡完成ResponseNet全部586条和seamless全部6,624条的生成、解码、绑定。
3. AV＋Event step8,927：同样完成两个全量数据集。
4. 仅AV step6,000：同上。
5. 仅AV step8,927：同上。

每个权重完成两个数据集后才进入下一个权重。对应权重的评测在独立机器并行接续，不阻塞下一权重生成。总计28,840个输出；每条保留生成MP4、独立listener WAV、模型latent和元数据。

评测机器固定172.29.189.119。轮询8卡，仅申请显存使用低于4GiB且利用率低于10%的空闲卡；正在忙的卡不抢占。每个已申请GPU最多4个分片进程，LPIPS/FID帧batch64，CPU检脸与读取并行；UTMOS使用该GPU的常驻独立环境模型。分片汇总、全局FID/FVD重算也在这台评测机执行。当前8卡均忙，评测队列等待空卡，三机生成独立推进。

## 输入与评分协议

- 固定两份sealed references和全部ID，标准输入仅speaker音频＋listener首帧。使用未微调Qwen2.5-Omni-3B；不加载规划LoRA，不将真实回应文本、listener后续画面或音轨作为生成输入。
- 规划使用现有SFT system/user指令，贪心最多512新token，无EOS时单条重试1024。原始输出与token保存在 `conditions/<dataset>/<id>/qwen/plan.json`。
- 两数据集均512×512。ResponseNet首帧与评分GT统一PIL bicubic，原GT文件不改，绑定传 `--size 512`，保留原FPS与帧数。seamless为512×512、25fps。12条ResponseNet帧格时长与声明音频窗口相差最多0.06秒，记录实际差值，保留原时间语义。
- 各权重共用规划及文本条件，每条seed137，原生30步Euler、video CFG3／audio CFG7。Native VAE所需8n+1帧在导出时裁回GT帧数。
- 九项主表：PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp、UTMOSv2。复用OmniResponse公式；ResponseNet face padding64，seamless为0。
- FID/FVD汇总全部分片特征重算，不平均分片分数。UTMOS沿用GT连续语音≥3.01秒、生成语音覆盖≥50%、两侧裁静音后>3秒规则，保持fold0；短回应／静音为N/A，保留覆盖率和漏回应统计。
- 音频评分只读独立生成listener WAV，使用 `--stages audio --quality-kinds utmos --audio-device cuda:0`。跳过整段视频解码和SyncNet／GLARE，不修改默认完整评测入口的原行为。
- `UTMOS_COMMON_WINDOWS.json` 在四权重均结束后按共同ID／时间窗口重新汇总；各权重原始覆盖率单独保留。不会以低覆盖率下的高均值替代全量表现。

## 状态、恢复和输出

[推理实时状态](status.json)、[评测实时状态](metrics_status.json)、[固定协议](protocol.json)、[推理配置](cluster.json)、[评测配置](metrics_cluster.json)。两个后台控制器的PID与命令分别在 `launch.json`、`metrics_launch.json`；日志为 `queue.log`、`metrics_queue.log`。

入口 `scripts/p2_testsets_eval.py`：

```bash
# 仅在对应旧控制器退出后恢复，完整输出会跳过
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_testsets_eval.py queue
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_testsets_eval.py metrics_queue
```

`generated/<arm>/<dataset>/<id>/` 保存MP4／WAV及元数据；`maps/`含显式generated_audio字段；`bindings/`与`metric_manifests/`保存完整绑定和24份评分清单。`metric_shards/`保存逐条分片结果和FID/FVD原始特征，`metrics/<arm>/<dataset>/summary.json`是最终九项及覆盖率。`RESULTS.json`是总表。

`SMOKE_COMPLETE.json`仅表示三机生成验证通过；`METRICS_SMOKE_COMPLETE.json`表示独立评测机的视频／音频链路验证通过；`GENERATION_COMPLETE.json`表示28,840条生成完成；`COMPLETE.json`表示评分已汇总，仍需查看partial和各指标覆盖率。不可把N/A当作0或高分。

## 修复与验证

原四机首次验证在导入PyAV时失败，原因是将Python3.10的评测依赖目录注入了Python3.13的生成环境。现已分离PYTHONPATH；原失败日志保留，未生成正式结果，不计入新队列。

本次41项现有回归＋6项调度／尺寸／特征汇总／UTMOS选择检查通过；三机24卡48条长样本生成和导出全通过，实测24卡利用率99%–100%。评测机空卡出现后先执行真实预训练UTMOS自对比和两数据集生成结果链路验证，再计算全量。GPU评测尚未发生，不声称已通过。

全量读取补充：586条ResponseNet WAV头与冻结清单核对后发现两条尾长量化差异，16kHz下分别多2个采样点、少8个采样点；仅尾部裁2点／补8个零，不改变时间原点或有效音频内容。读取允许ResponseNet至多1ms尾部量化差，超出仍报错，seamless规则不变。两条真实读取复验通过，详见 [AUDIO_ENDPOINT_CHECK.json](AUDIO_ENDPOINT_CHECK.json)。

全量缓存实测配置：Qwen规划batch16、Gemma batch16。原生Gemma batch32在全层float32后处理中超过139.81GiB显存，现使用未改公式的原生batch16。Qwen结果及已完成小批量输出复用；分块后处理候选未通过逐位一致检查，未用于任何提交的缓存，最终仍调用原生encode_batch。

评测恢复记录：ResponseNet小批量音频评分曾因构造清单漏source_start失败；已补齐source_start／source_end，两个真实GT／生成音频读取复验通过。失败报告另存，评测控制器已重新启动，继续等待空卡。该问题仅在小批量清单，正式绑定仍使用原完整字段。
