# 显式规划缓存三机配置验证

- [三机运行说明](../../related/P2_显式规划缓存三机联合运行.md)
- [配置](../../configs/p2_plan_cache_3nodes.json)
- [最终验收与源码哈希](validation.json)：71项测试通过；[JUnit](pytest.xml)。
- [实际三机24分片烟测](smoke24_final/smoke_validation.json)：三个阶段、257个虚拟ID，逐阶段无重复/遗漏，未使用GPU。
- [实际运行状态](smoke24_final/status.json)：三个远端监督进程与24个分片均成功结束。
- `smoke24_final/preflight/`保存三机SSH、双向共享目录、模型/代码依赖与GPU探测。

初次`smoke24/`记录了调试期间启动返回码的问题（子任务已正常完成），已修复并重跑。最终结果只以`smoke24_final/`与`validation.json`为准。

探测时24张GPU均有其他计算进程；未启动正式缓存，未停止其他任务。既有单字中文数据约束仍须在全量prepare前解决。
