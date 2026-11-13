# step1500后切换到完整全attention／FFN LoRA训练

用户授权：当前去Event训练保存step1500后停止，转入已准备的完整pipeline GPU测试并训练。

- 旧组：`../p2_current_component_ablations31_20261108/no_event`，在status达到1500、进入同步保存时写入原生STOP；等待完整保存和全部31rank退出，再验证模型、Adam、31份RNG及SHA。
- 新组：本目录上级配置不变，从预训练独立初始化，rank8／alpha8，原生attention及FFN LoRA，两路Whisper attention全参数，AV＋0.05整段Event＋两路各0.025 Feature；最终284218段／500.001259h及最新caption。
- 物理分配沿用当前训练：node12／node8／node4 GPU0–7，node11 GPU1–7；保留node11 GPU0现有任务。仅独立启动wrapper和cluster物理映射改变，模型／数据／损失源码不变。
- 首先执行最长11.95秒样本两次真实更新并恢复初始状态，然后训练两步保存step2，CPU全量核验及31rank退出后严格从step2继续。
- 正式预算沿用已准备配置9169步，每500步保存；全重计算、CPU预取2、512MiB FP64归约、GPU Adam。
- 任一GPU测试或检查点验证失败，阻止正式启动并写ERROR.json；不覆盖旧权重、不终止非本组任务。

状态：STATE.json；旧组保存验收：OLD_STOPPED_ACCEPTED.json；GPU验收：GPU_DEBUG_ACCEPTED.json；正式启动：FORMAL_STARTED_ACCEPTED.json；速度：PERFORMANCE.json。文件仅在对应阶段真实完成后产生。

## 实际启动验收

旧组精确停止于1500，37,037,039,487字节完整检查点SHA、模型／Adam及31份RNG验收通过。新组最长样本全31rank测试最高99.291015625GiB，step2完整保存37,833,890,366字节并校验通过，随后严格恢复并实际完成第3步之后的更新。五个优化器组均有有限非零梯度，损失加权公式一致。正式启动证据为FORMAL_STARTED_ACCEPTED.json，当前步数见上级status.json。
