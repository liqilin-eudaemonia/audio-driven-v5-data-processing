> 已由用户改为三机24卡逐权重推理，以下保留历史。当前运行见[新队列](../p2_testsets_4ckpt_24gpu_20261102/README.md)。原队列首次GPU缓存验证因PyAV环境混用退出，无正式生成结果。

# 四机32卡：四组权重、两个完整测试集

2026-11-02，按用户要求排在当前仅AV训练完成之后。实时状态见 [status.json](status.json)，后台控制器日志为 `queue.log`。GPU链路尚待训练释放资源后验证；只有 `SMOKE_COMPLETE.json` 生成才表示32卡小批量验证通过，只有 `COMPLETE.json` 才表示全量推理及评测汇总结束。

## 固定协议

- ResponseNet全部586条；seamless全部6,624条。两份原始 references.jsonl 与 seal 保持不变。
- 两个数据集均生成512×512。ResponseNet参考首帧和评测GT由原640×640使用PIL bicubic缩到512×512；原始GT文件保留，绑定时显式传 `--size 512`，评分清单记录 `gt_resize` 和 `gt_resize_method`。seamless使用最终裁框及原512×512输出。
- ResponseNet保留各样本原帧率，seamless为25fps；输出帧数按对应GT。12条ResponseNet的帧格总长比清单声明时间多至0.06秒，保留GT帧数和原声明的音频时间窗，差值记录在每条 geometry.gt_frame_grid_offset_s，不重新对齐或删样本。
- 标准输入仅speaker音频＋listener首帧。未微调Qwen2.5-Omni-3B自主生成回应描述，不加载规划LoRA，不将目标描述、listener未来视频或listener音频输入模型。
- 使用现有 `p2_planning_sft.data.SYSTEM/USER` 指令；贪心生成最多512新token，未EOS时单条重试1024，仍未结束则报错。每条原始规划与生成token保存到 `conditions/<dataset>/<id>/qwen/plan.json`。
- 生成规划通过当前J的 `caption_prompt()` 同时用于Qwen hidden与Gemma文本条件，四组共享同一份规划及条件缓存。
- AV＋Event和仅AV各选step6,000、最终step8,927，共4组、28,840个输出；路径与源清单哈希见 [protocol.json](protocol.json)。不改变当前训练预算。
- 每条固定seed137；原生30步Euler，video CFG=3、audio CFG=7。生成符合VAE的8n+1帧，再裁到GT帧数；音频保留到原时间窗。

## 调度与提速

控制器等待两组训练status.complete且step=8927，验证四份检查点哈希，等待四机32卡显存释放。首先每卡两条，覆盖两个数据集，共64条×4权重=256个输出，完成cache→render→decode及两数据集真实指标链路验证；通过后自动全量运行。验证输出直接复用。

32卡每卡独立进程，按原生生成帧数分配任务。Qwen规划batch16、Gemma batch32，条件缓存供四组权重共享。DiT在每张卡内跨样本／权重常驻；参考VAE也常驻。视频、音频VAE解码独立阶段运行，避免与DiT争显存。完整输出标记支持失败后续跑；每个工作进程最多自动重试3次，持续错误会写status.failed并停下。

代码入口：`scripts/p2_testsets_eval.py`，实现位于 `src/p2_testsets_eval/`。手动恢复命令（确认旧控制器已退出后）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u   /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_testsets_eval.py queue
```

## 输出与指标

`generated/<arm>/<dataset>/<id>/generated.mp4`、`listener.wav`、`metadata.json` 为生成结果；`bindings/`保存官方绑定检查，`metric_shards/`保存32卡分片逐条评测，`metrics/<arm>/<dataset>/summary.json`与`RESULTS.json`保存最终结果。

复用OmniResponse既有八项视频指标公式：PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp。分片导出原始特征，最终全量重算FID/FVD，不平均分片距离。像素和LPIPS按有效帧数合并，表情与ΔResp按原有公式汇总。脸部检测失败等情况记录实际覆盖率，不填补分数；八项指标不包含音频质量评测。ResponseNet face padding=64，seamless=0，保持原评测约定。

## 已完成验证

既有25项指标／适配器回归通过；新增4项分辨率、分片与全局特征汇总检查通过；新增ResponseNet显式512绑定检查通过。两个数据集各长短样本共4条真实媒体读取通过，全部得到512×512首帧，详见 [CPU_MEDIA_CHECK.json](CPU_MEDIA_CHECK.json)。现有三份候选权重的源代码、模型资产和877个参数形状核对通过，见 [CHECKPOINT_COMPATIBILITY.json](CHECKPOINT_COMPATIBILITY.json)；仅AV最终权重将在完成后检查。以上CPU检查不能替代尚未执行的GPU完整链路验证。
