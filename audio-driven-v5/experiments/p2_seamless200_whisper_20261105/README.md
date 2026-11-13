**当前已完成：** 两权重共400条生成、全部指标计算及HTML接入结束。现有12组／16条典型视频，新增32个Whisper媒体项；浏览器音轨、切换、排名与移动布局验收通过。完整结果见`RESULTS.json`、`RESULTS.md`和`FINAL_RUN_AUDIT.json`。每组1条人脸／关键点覆盖不足，UTMOS无有效窗口按N/A保留；音频执行错误0。下文等待队列描述为启动时记录。

# Seamless200 Whisper speaker cross-attention补测

范围：Whisper AV的6000／8000步权重，各推理固定Seamless200，共400条。等待用户授权的28卡释放后自动开始，不停止现有任务。资源是node17、node12、node4各8卡，node8（172.29.104.237）仅0–3卡。

执行状态见`status.json`；最近资源检查见`resources.json`；CPU预检见`CPU_PRECHECK.json`。CPU预检不代表真实GPU端到端已通过。GPU完成后自动进入指标队列，最终写`RESULTS.json`、`RESULTS.md`、`COMPLETE.json`。

## 对照协议

- 同一200条test、最终512×512修正版裁框、原始speaker音频＋listener首帧。GT listener净化音轨仅用于评分，GT描述和未来listener画面不作为输入。
- 复用十组评测的原生Qwen2.5-Omni-3B规划文本、H、Gemma缓存和空文本条件；不更换为SFT Qwen。
- 额外Whisper-large-v3 encoder仅处理speaker，FP32、batch16、50Hz，时间从当前pair起点0秒开始，保留全部静音。
- 原Qwen H保留；训练好的Whisper独立视频／音频cross-attention残差分支同时保留在CFG正负路径。
- 复用原十组采样函数，seed137、30步Euler、videoCFG3/audioCFG7、512×512/25fps、原片段导出时长。视频初始噪声FP32，与原十组一致。
- 6000与AV6000是主要对照；8000仅观察进展，不能作为与AV8927等预算实验。
- 使用原评测公式和GT特征／完整RGB解码缓存；FID/FVD合并全200条特征，报告情绪／运动／短回应音频的有效覆盖率及N/A。

## 调度与速度

一卡一常驻进程；28份任务按视频token量均衡分配，两权重在同进程内依次加载，复用renderer和VAE。Whisper先批量缓存，随后生成、解码，再用GPU检测／表情模型计算指标。GT无损解码缓存和已有原生Qwen条件均复用，避免重新生成规划、反复解码GT。

完整训练权重保持不动；预检逐位核对模型参数后导出仅含模型状态的推理副本，省去每卡读取优化器。每条输出原子提交，重复启动跳过完成项。`controller.lock`防止重复控制器；资源每30秒检查，连续两次28卡均显存<2048MiB、利用率<10%才启动新GPU阶段。SSH不可达时等待并记录，不视为空闲。不使用node8的4–7卡。

## 命令与代码

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_seamless200_whisper_eval.py queue \
  --work experiments/p2_seamless200_whisper_20261105
```

入口：`scripts/p2_seamless200_whisper_eval.py`；匹配采样与缓存：`src/p2_testsets_eval/whisper_runtime.py`；回归：`tests/test_seamless200_whisper_eval.py`。不修改冻结的`src/p2_whisper_av/`和私有LTX源码，启动时校验权重与代码身份。放置`STOP`可阻止后续阶段，不会终止其他任务。

推理副本的`checkpoints/*.export.json`保存原权重SHA256、推理副本SHA256及参数逐位一致核验。推理输出metadata继续记录原训练权重来源。当前排队期间没有可承诺的开跑时间；首批真实GPU耗时出来后才能估计剩余用时。


已完成CPU验收：新增匹配采样／CFG／speaker时间轴／资源范围回归3项，既有评测与队列回归13项，共16项通过。两份模型推理副本全部参数逐位一致。后台控制器已启动，PID见CONTROLLER.json；当前状态为等待授权28卡释放。GPU端到端尚未执行。
