# LTX 主干 Event 验收

87 项 CPU／原生小规格 LTX 测试通过。完整 284218 条 train＋dev 元数据预检查、当前五项混合缓存回读、原生 BF16 坐标与 FP32 标签时钟核对通过。独立全量区间审计：越界区间 0、真实片长不匹配 0、静音／发声冲突 0。

- 汇总：VALIDATION.json
- 测试：tests.log
- 全量预检查：preflight.json
- 实际缓存：CPU_CACHE_CHECK.json
- 独立区间审计：INDEPENDENT_INTERVAL_AUDIT.json
- 旧代码校验：LEGACY_RUNTIME_INTEGRITY.json
- 32 卡参数：LAUNCH_COMMANDS.json
- 采样完整性：SCHEDULE_CHECK.json

新训练未启动；22B 完整 GPU 显存、吞吐和生成质量不在本轮已验证范围内。
