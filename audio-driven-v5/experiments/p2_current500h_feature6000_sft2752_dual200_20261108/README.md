**后续已接四机32卡：Whisper6000保存后迁移扩容，保留本目录所有已完成结果和GPU7当前样本。当前单卡推理继续，四机尚待触发。[统一接续](../p2_current_feature_whisper6000_eval32_20261108/README.md)。**

# 当前500h AV＋Feature step6000＋SFT2752双200评测

接续当前纯AV6000的评测，使用node11物理GPU7，不停止其他任务。Seamless固定200＋ResponseNet分层随机200，共400条。

## 权重与训练对照

- 权重：`../p2_current500h_av_feature31_20261107/step_006000.pt`。
- SHA256：`9049cd752edb21069476bc69d68bcac77c64c4e5cb93ad196e2dd5af7f6a6e84`。
- 原生schema：`p2-current500h-qwen-av-feature-v1`，完整模型、Adam及31份rank RNG。严格核对保存的训练源码身份，加载原生Feature Model，保留两路Feature Decoder，禁用Event／Whisper。推理只调用Connector和AV生成，**不运行Feature loss，不读取teacher／未来listener媒体**。
- 训练损失：AV＋0.025视频Feature＋0.025音频Feature，Event=0。与当前纯AV6000使用相同500h选集、31卡batch、训练seed、样本／noise调度、Connector、Qwen和text LoRA配置，可作为Feature单项对照。

## 共享推理条件

与刚测AV-only及联合6000／6500读取同一400条最新SFT2752规划文本、H、Gemma和首帧；训练H仍是原生Qwen，推理H为SFT2752，来源保留。无需再次生成规划或提取条件。

seed137、Euler30、视频CFG3／音频CFG7，512×512及每条原FPS／原帧数保持。没有speaker Whisper分支。GT和音频评分规则沿用原测试协议。

## 执行与资源

入口：`scripts/p2_current_feature6000_sft2752_eval_20261108.py prepare|queue|worker --work <本目录>`。仅node11 GPU7；渲染启动前连续两次确认可用显存≥80GiB，可与已有任务同卡运行，不杀其他任务。父进程通过锁防止重复，阶段失败最多3次，完整样本沿原生逐样本锁及complete标记续跑。

顺序：单进程生成 → 4进程同卡原生解码 → 绑定 → 8个评分分片，使用CPU预取／像素流水线 → 合并全200特征计算FID/FVD → 最终逐条验收。评分沿用显存余量准入，预留16GiB，新增每worker预算8GiB，最多8路；不再因其他任务存在而一直等待空卡。

CPU_ACCEPTANCE.json：权重SHA／有限性／模型与Adam、31份RNG、源码及400条条件复用通过。RUNTIME_ADAPTER.json逐项记录与原AV评测入口的差异；AV数值采样函数不变，真实GPU严格加载和首批生成以GPU_MODEL_ACCEPTANCE.json及progress/render/07.json为准。2项回归测试验证同训练／输入控制及采样函数未改。

## 状态与产出

- 实时阶段：status.json、CONTROLLER.json、GPU_WAIT.json、controller.log、progress/。
- 最终指标：RESULTS.json、metrics/；最终验收：FINAL_AUDIT.json和HANDOFF_COMPLETE.json。
- COMPARISON.json同时读取当前AV-only6000和联合6000，便于Feature单项对照及联合参考。
- 完整媒体：generated/current500h_av_feature_6000_sft2752/{seamless,responsenet}/<id>/。

本目录STOP停止继续领取和后续阶段，不影响其他任务。共享条件目录保留在p2_ltx6000_latest_sft_dual200_20261107，不要删除。代码快照及SHA存于code_snapshot与CODE_SHA256.json。没有恢复Goal或聊天定时巡检。
