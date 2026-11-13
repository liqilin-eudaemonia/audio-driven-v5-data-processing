# 全attention／FFN LoRA6500：ViCo正式test43

**已完成并通过独立复核（2026-11-11）：** 43条生成、86个视频副本解码核验、官方五项＋运动补测＋DIM十二项完成；独立方差／BFM FD及公开DIM代码对照通过。[完整结果](RESULTS.md) · [43条视频／音轨](VIDEOS.md) · [数值复核](FINAL_USER_REVIEW.json) · [执行修复复核](FINAL_EXECUTION_REVIEW.json)。

用户在完成6000步ViCo评测后，要求继续测试6500。本轮沿用同一test43、SFT2752条件、seed137、Euler30、CFG3／7、512×512／25fps生成后转256×256／30fps评分，仍仅node11物理GPU7。最新要求为每小时巡检、持续九小时，取代此前每20分钟；[本次及ResponseNet巡检](../p2_vico_responsenet_watch9h_20261111/README.md)。

## 权重来源

原6000目录没有6500；同一完整模型已从6428迁移至batch28继续训练。使用：

`/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_all_lora_full28_batch28_20261109/step_006500.pt`

原始SHA256：`4bf7a44137130488fd60c0c2c5a16eae495dc6ce4fc8c2024cee212de2232a52`。MIGRATION记录证明6428迁移时模型和Adam逐值相同；当前6500模型、Adam和28份RNG完整，295份训练来源核对通过，1632处rank8 LoRA及Whisper全参数、Event0.05／双路Feature各0.025保持。

复用已有batch28 checkpoint loader与配置规范化：原始训练身份保留，按已验证的同模型配置构建并严格加载，不改写其batch28来源。独立进程显式选择用户新授权的6500，不修改历史5500／7000／7500队列的白名单或恢复其任务。

## 加速与评测

- 新导出不含Adam的模型推理包，逐张量与完整6500权重相同；GPU只读mmap，完整checkpoint只在CPU导出时校验。
- 直接复用6000一轮相同的原始ViCo SFT2752规划／H／Gemma／speaker Whisper／首帧与负条件缓存；不读取未来listener媒体或GT描述作生成输入。
- 沿已跑通6000的原生采样与转换入口，生成和解码分阶段；最多4个互斥解码分片按显存启动，保留16GiB余量。6500解码在整分片外再沿用现有VAE loader缓存，避免恢复latent路径逐样本重载；CPU CPBD与解码重叠。
- 官方画质／身份5项、CustomListener风格运动补测、DIM Exp／Pose十二项全部接续；不跑AF SPECTRE。GT／speaker特征复用，生成侧重新提取。
- Var仍按用户指定的逐系数总体方差平均，原DIM拉平方差另存；单次seed137，不做GT择优。

## 状态、代码和结果

[CPU来源及导出核验](CPU_ACCEPTANCE.json) · [启动回执](LAUNCH.json) · [阶段状态](status.json) · [实际生成进度](progress/render/00.json) · [真实GPU模型加载](MODEL_ACCEPTANCE.json)。latent阶段status是阶段快照，完成数以实际`latents/all_lora6500_sft2752/vico/*/complete.json`为准。

推理入口：`scripts/p2_vico43_all_lora6500_gpu7_20261111.py`；运动／DIM入口：`scripts/p2_vico43_all_lora6500_metrics_20261111.py`。6000源脚本、封存结果及视频保持不变。

完成后生成`RESULTS.md`、`ALL_RESULTS.json`、`IMAGE_RESULTS.json`、`bfm/RESULTS.json`、`DIM_RESULTS.json`、`FINAL_AUDIT.json`及`COMPLETE.json`。原生视频在`generated/all_lora6500_sft2752/vico/<id>/generated.mp4`，评分副本在`scoring/all_lora6500_sft2752/`。没有最终43条验收时不称完成。

与6000比较须保留6500来自6428后batch28续训的说明；评测设置相同，不把训练批量变化隐去。

运动抽取曾被未参与对齐的辅助68点差异中断；独立适配器保留原失败诊断，仅以实际五点／RGB裁图一致及原系数容差判定，原网络和系数／评分数学不改。小显存准入用于同卡仅余约23GB的情况，最多两路8GB上限；原始媒体和已完成系数保留。详见`BOUNDED_MOTION_POLICY.json`及`FINAL_EXECUTION_REVIEW.json`。本轮DIM有效16,237帧，6000为16,238帧，各自掩码保留。
