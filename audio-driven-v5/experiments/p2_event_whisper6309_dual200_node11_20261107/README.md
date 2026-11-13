> 当前已扩为node11八卡：rank0沿用GPU7原renderer，rank1–7在GPU0–6；新统一controller接管后续阶段。单卡信息为历史，当前启动与状态见 [八卡扩容说明](expansion8/README.md)。不要重启旧单卡入口。

> 已通过真实启动验收：仅GPU7，6309步模型严格加载成功，首条11.92秒样本完成推理，视听latent均有限、来源及seed／CFG校验通过；即时GPU利用率100%，显存约52.3GiB。见 `STARTUP_ACCEPTANCE.json`。当前后台继续生成，其余解码和指标自动接续。

# AV＋Event＋speaker Whisper：6309步，双200评测

本轮仅使用node11（172.29.189.119）物理GPU7。用户允许与轻量任务叠加，不停止其他任务。训练保持停止，不启动Feature训练。

- 模型：`experiments/p2_whisper_recompute30_adopted_20261106/event/step_006309.pt`；原始Qwen H＋AV＋0.05 Event＋speaker Whisper，不含Feature，规划器为未微调Qwen2.5-Omni-3B。
- ResponseNet：既有分层随机200条，`experiments/p2_responsenet_random200_twelveway_20261105/references/references.jsonl`。
- Seamless：既有固定200条，`experiments/seamless_quick200_latest_20261105/prepared_200/references.jsonl`。
- 共400条生成；使用相同seed137、Euler30、video CFG3／audio CFG7、512×512原生帧网格及历史首帧／speaker音轨输入。
- 只输入speaker音轨和listener首帧，不输入GT未来画面、listener真值音轨或GT打标描述。400条Qwen/Gemma／speaker Whisper缓存从上一轮复用，准备时逐条检查来源和输入审计。
- 6309步为首4000步batch31、后2309步batch30，共193270次样本展示；历史32卡6000步为192000次，本轮多约0.66%。不能宣称严格相同步数／优化轨迹。

## 运行与加速

入口：`scripts/p2_event_whisper6309_eval_node11_20261107.py`。独立实验目录，原历史结果及检查点不覆盖。模型只导出推理需要的state，CPU逐张量校验，原训练检查点保留Adam和30份RNG。

常驻完整模型串行处理400条，不反复加载各片段；VAE encoder与decoder沿用已有常驻缓存。评分复用无损GT解码及FID/FVD／人脸特征缓存，batch64、face/emotion CUDA、pixel workers4；拆8个均衡评分分片，每卡4个评分进程，始终限GPU7。FID/FVD汇总全200条特征计算，不平均分片分数。

自动阶段：CPU权重导出和输入校验 → render → decode → bind → 双数据集实际生成与评分冒烟 → 原生视频／音频指标 → 全量汇总与验收。

PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp及既有音频活动／质量指标沿用 `evaluation/README.md`。保留人脸不足、静音和短回应的覆盖率与N/A，不能将其填零混入均值。

状态：`status.json`、`progress/render/00.json`、`progress/decode/00.json`；评分阶段另看 `metrics_status.json`。权重导出在 `checkpoints/step_006309.export.json`，输入准入在 `CPU_PRECHECK.json`，实际GPU身份在 `device.json`。全部结束以 `FINAL_AUDIT.json` 和 `HANDOFF_COMPLETE.json` 为准，结果在 `RESULTS.md`／`RESULTS.json`，逐条在 `metrics/`。

结果媒体：`generated/base_event_whisper_6309/{responsenet,seamless}/<sample_id>/{generated.mp4,listener.wav,metadata.json}`。

创建本目录STOP会在样本边界停止共享领取及后续阶段，完整产物保留；本轮评测STOP独立于训练STOP。默认不要重复启动活跃controller。确需续跑时，在node11上执行：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_event_whisper6309_eval_node11_20261107.py launch
```

历史参考：`experiments/p2_combined_whisper_eval_20261106/RESULTS.json`；相同两套测试池和采样协议。评测完成后生成 `COMPARISON.json`，本轮权重身份单独保存，不冒充历史联合模型。
