# 最新500h：AV＋LTX整段Event

用户指定下一组优先补齐单加Event消融。当前AV＋Whisper6000完整保存、校验并退出后，本组从原始预训练权重独立初始化，四机31卡训练至6000保存退出，再接原三组去组件实验。每500步完整保存，node11 GPU7预留。

目标为`AV loss + 0.05 × LTX hidden整段Event loss`；无Whisper分支、无EmbeddingDecoder或Feature loss，不打开speaker Whisper或listener teacher缓存。Event沿用当前完整模型最后LTX block的有效生成音视频hidden、整段聚合、原19类配置及18类有效mask；正例、冲突／未知屏蔽和0.1弱负例权重不变，无时序Event监督。

最终284,218段／500.001259小时，train281,060＋dev3,158、test0，最新完整caption及现有Qwen／Gemma／AV和Event数字缓存；同31卡样本／noise调度。原始Qwen hidden冻结，优化Connector、LTX text LoRA、Event头。

独立模块`src/p2_current_av_event_only`及`scripts/p2_current500h_av_event_only.py`，复用已验收组件，不修改原三组或当前训练冻结源码。[本组配置](config.json) · [验收](ACCEPTED.json) · [真实缓存回读](preflight.json) · [6000步启动命令](COMMANDS_STOP6000.json)。实际小模型更新／Event独立梯度／冻结主干／完整Adam下一步严格恢复／全重计算loss与全部梯度一致／禁用输入与缓存拒绝测试通过；32条真实缓存回读通过；冷CLI到GPU门槛通过。GPU完整规模显存与NCCL验收在实际分配资源后执行，最长样本两次完整更新并恢复初始状态后才进入正式训练。

[新队列入口](../sequence_v2.py) · [队列授权](../SEQUENCE_V2_AUTHORIZED.json) · [实时队列状态](../SEQUENCE_STATE.json)。顺序：当前AV＋Whisper→AV＋Event→去Whisper→去Feature→去Event；每组6000完整保存校验、原rank退出及空卡检查通过才接下一组。不要重复手动launch。
