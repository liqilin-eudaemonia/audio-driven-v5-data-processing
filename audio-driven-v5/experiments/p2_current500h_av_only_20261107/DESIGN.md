# 最终500h上的原始Qwen AV-only对照

用户已确认：沿用最早Qwen条件、不加Whisper的AV-only pipeline；仅替换为最终train+dev 284218段/500.001259h及最新caption对应的Qwen H和Gemma缓存。代码先备份，实验只准备，不自动启动GPU。

沿用原始未SFT Qwen2.5-Omni-3B H、首帧条件、配对speaker audio、指导文本、512px AV VAE目标和净化listener音轨。训练Connector和LTX文本LoRA；不读Whisper、teacher、事件标签，不执行辅助损失。沿用旧AVOnlyModel初始化顺序，冻结未使用的历史Event头以保持初始化兼容；优化器仅包含实际可训练参数。

默认32卡/batch32，从预训练原始初始化开始，seed20261023，复用本轮联合训练最初的8882步/时长512调度（同样本和noise ordinal），每500步保存、all全重计算、CPU预取2、512MiB FP64归约、GPU Adam。正式机器安排未授权，本轮不占卡。CLI也可显式准备24卡独立配置，但不将24/32卡轨迹宣称完全相同。

采用独立p2_current_av_only包与CLI，不修改当前运行绑定的270份源码。拒绝加载联合训练/旧数据实验作为初始化或resume；续训必须属于本实验的完整检查点。复用已完成的全量缓存准入记录，全量核对样本顺序、split、时长和调度，抽样验证当前caption/缓存/条件/原生AV输入，CPU微型LTX实测反传及完整checkpoint恢复。正式GPU速度和最长样本验证留在用户安排机器后的启动阶段，首次训练前强制进行最长样本前后向。
