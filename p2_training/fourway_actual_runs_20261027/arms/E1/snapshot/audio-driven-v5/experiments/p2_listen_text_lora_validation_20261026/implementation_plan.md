# 已授权的 text-attention LoRA 实施方案

基于 GitHub c72c10a 的无 feature 版。用户已选择两组同时加入 text-attention LoRA。

- 复用 v4 P2.1 注入器：每个 block 的 attn2/audio_attn2，Q/K/V/out；rank=8、alpha=8、LR=1e-5 作为初始配置。
- 原生权重冻结；Connector、事件头与 LoRA 由同一 DDP 根持有。AV 前向放入该根 forward，事件头仍读取 U，事件项不直接到 LoRA。
- LoRA 独立优化器组；Connector 与 LoRA 作为生成参数共同裁剪，与事件头分开。保留 FP64 累加/同步。
- 检查点仅增加 adapter 权重及优化器状态；身份包含目标层列表和复用代码。恢复和生成严格重建 LoRA。
- remove_u 保留已训练 LoRA；text_only 使用原生 LTX。正常与音频替换推理的 CFG 两分支共享同一套 adapter。
- 新配置和输出目录；原无 LoRA 配置不变，feature 保持关闭。冻结缓存指纹源、v4 源码与正在运行的全量抽取配置不修改。
- 验证：CPU 参数所有权/梯度/精确恢复/推理加载，真实原生 attention CPU 小模型，两进程参数一致性，真实缓存复用；有空卡时执行真实 GPU S0。正式效果训练不自动启动。
