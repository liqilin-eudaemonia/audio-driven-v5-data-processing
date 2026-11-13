**本组400条正式评测已全部完成并通过FINAL_AUDIT；指标已加入原总结果页。** [查看同SFT条件下的当前AV基线与联合模型对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。构建及浏览器验收见dashboard_update/。下方等待／叠加安排保留为历史执行记录。

**最新安排（2026-11-08）：已获用户授权叠加评分。只使用node11 GPU7，按显存余量最多8路评分，预留16GiB，不再等待完全空闲。400条生成／解码及smoke保留。下文“等待完全空闲”为历史策略；见[叠加评分说明](overlay_scoring/README.md)。**

# 当前500h AV-only step6000＋最新SFT2752：双200评测

用户确认使用最新SFT2752，与刚测联合6000／6500共享完全相同的400条规划文本、H、Gemma、speaker输入审计和listener首帧。LTX为独立AV-only6000，不读取Whisper特征，不包含Whisper注入、Event loss或Feature loss。训练H来自原生Qwen，推理H来自最新SFT，来源差异明确保留。

## 权重与协议

- 权重：`../p2_current500h_av_only31_20261107/step_006000.pt`，1391207747字节；SHA256 `77f37c9dd7ab69f24af5d700454e2a8a5ed2ce4f81f7216b12a282822977f0a3`。
- 独立schema `p2-current500h-qwen-av-only-v1`，使用原生Model和严格加载，不能套用joint/Whisper loader。完整Adam与31份RNG保留，评测不恢复训练状态或启动训练。
- ResponseNet分层随机200＋Seamless固定200，seed137、Euler30、视频CFG3／音频CFG7、512分辨率及每条原FPS／帧数保持。
- 数据与缓存指向`../p2_ltx6000_latest_sft_dual200_20261107/conditions/`，勿删除共享来源；不重新生成Qwen规划或提取H／Gemma。

## 等待与执行

控制器已挂到node11，唯一授权GPU是物理7号。每30秒检查一次，连续两次满足显存<2048MiB、利用率<10%、且没有compute进程才开始；即使另一个任务仅占少量显存也不叠加。仅为该作业的空闲接续，不恢复Goal或聊天定时巡检。不会停止其他任务。

顺序：单进程原生AV生成400条 → 同卡4进程原生VAE解码／编码 → 原生绑定 → 同卡8个评分分片，使用已验证的CPU预取／像素计算流水线 → 全200特征合并FID／FVD → 最终逐条验收。音频评分保留静音、短回应与UTMOS N/A规则。8路并发与CPU流水线的全量联合吞吐仍以本次实际运行为准。

入口：`scripts/p2_current_av6000_sft2752_eval_20261107.py prepare|queue|worker --work <本目录>`。queue必须在node11运行，CPU控制器可设置CUDA_VISIBLE_DEVICES为空，子进程固定7。文件锁避免重复控制器；阶段失败最多3次，保留成功样本后续跑，最终失败写FAILURE.json。创建本目录STOP会停止继续领取和后续阶段，不停止其他训练。

## 验收与状态

完整权重SHA、有限性、模型与Adam、31份RNG、当前训练源码身份、纯AV配置及400条SFT条件复用CPU检查通过；见CHECKPOINT_ACCEPTANCE.json、CPU_ACCEPTANCE.json。3项测试覆盖小占用任务也必须等待、GPU空闲门槛、固定条件／清单／GPU授权。GPU严格模型加载将在空闲后执行，以GPU_MODEL_ACCEPTANCE.json为准，排队不代表GPU验收或评分完成。

查看status.json、GPU_WAIT.json、CONTROLLER.json、controller.log以及progress/。最终以FINAL_AUDIT.json和HANDOFF_COMPLETE.json为准；指标RESULTS.json，联合6000参考对照COMPARISON.json。该对照同时改变Event、Feature、Whisper三个组件，只能用于组合pipeline比较。

代码快照与CODE_SHA256.json同目录保存，相关通用优化说明见[CPU流水化说明](../../related/P2_评测CPU预取与像素计算流水化提速_20261107.md)。
