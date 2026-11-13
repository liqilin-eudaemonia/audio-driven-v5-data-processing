# Seamless固定200：Whisper与联合6000的新seed配对复测

用户指定node11（172.29.189.119）仅GPU0。本评测不占用其他卡，不修改其他训练或停止标记。

- A：当前500h AV＋Whisper6000＋最新SFT2752；来源`../p2_current500h_whisper6000_sft2752_dual200_20261108`。
- B：LTX主干Event＋Feature＋Whisper6000＋最新SFT2752；来源`../p2_ltx6000_latest_sft_dual200_20261107`。
- 新seed为238；原seed137结果保持原目录。两组均使用固定Seamless200，512画面、原FPS、Euler30、videoCFG3／audioCFG7；同一seed用于各样本的原生采样逻辑。
- 本轮只改变LTX生成seed，SFT2752规划文本、H、Gemma、listener首帧及speaker Whisper缓存完全复用。没有重新生成Qwen规划，不读取未来listener或GT caption作为输入。
- 各组独立目录`A/`、`B/`，与旧seed的输出路径完全不同；CPU验收及3项测试通过。加载与生成沿用各自已验收的原生模型／冻结来源；不修改正在训练的源码或配置。
- 单GPU顺序评测A、B：每组常驻生成模型→最多4路共享VAE解码（OOM降为2）→最多8路评分。沿用CPU预取2、像素线程4、batch64、GT RGB／特征缓存和完整200条FID/FVD汇总；静音与过短音频按原口径报告N/A和覆盖率。
- 入口：`scripts/p2_seamless_whisper_joint_seed238_20261108.py prepare|queue|worker`；状态`status.json`、`A/progress/`、`B/progress/`；评分`A/metrics_status.json`、`B/metrics_status.json`。PID与实际卡号在各目录`processes/`、`devices/`。
- 原生输出在`{A,B}/generated/<arm>/seamless/<id>/generated.mp4`与`listener.wav`。各组结果`{A,B}/RESULTS.json`；两组完成后生成根目录`RESULTS.json`、`SEED_COMPARISON.json`和`FINAL_AUDIT.json`。最终验收必须400条完整，seed238与原始权重／SFT身份匹配，全部200条核心视频指标完成且音频技术错误0。
- 仅占用GPU0，显存不足时不启动，不停止别的任务；未恢复Goal或聊天巡检。根目录STOP停止后续领取，若正处评分阶段可同时写入当前`A/STOP`或`B/STOP`以停止其评分队列。

seed137与238用于观察结果对生成随机种子的敏感性；两次点估计不能代替多seed显著性检验。
