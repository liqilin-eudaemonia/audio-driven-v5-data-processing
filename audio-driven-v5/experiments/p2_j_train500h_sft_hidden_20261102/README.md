# 当前 500h：规划 SFT Qwen guided H

已完成：285,638／285,638条新规划SFT guided H，96个worker均正常退出、失败0，全量独立验收通过。32进程最终审计耗时24.25秒，结果见 COMPLETE.json 和 completion_summary.json。四机32卡现已释放，未启动训练。范围为原 285,638 条 train，500.00028h。新特征单独保存，原标注文本、Gemma、AV latent 和 teacher 继续复用。

完整说明：[数据、权重、优化、验收与读取交接](../../related/P2_规划SFT后500h隐藏特征接续提取_20261102.md)。

| 文件 | 作用 |
|---|---|
| `protocol.json` | 冻结成员、原缓存、最终 adapter SHA、提取源代码身份 |
| `CPU_CHECK.json` | 32 条真实 CPU 输入核对，8 条原视频 processor 对照 |
| `cpu_tests.xml` | CPU 回归测试结果 |
| `node_preflight.json` | 四机 SSH、Python、源码及 adapter 可见性核验，不加载 GPU |
| `chain_launch.json` / `chain_status.json` | 后台接续器 PID、当前阶段及异常 |
| `execution.json` | 冻结的四机执行配置 |
| `CANARY.json` | 当前协议的 GPU 小样本验收；不存在时尚未通过 |
| `cache/input_plan/<sample_id>/` | 新规划 SFT Qwen 的 payload 与身份校验文件 |
| `cluster_status.json` / `nodes/` | 全量进度、节点日志及 GPU 利用率采样 |
| `COMPLETE.json` | 全量独立验收；不存在时不能视作完成 |
| `source_snapshot/` | 本轮新增代码、测试、配置和依赖指纹快照 |

前序缓存：`../p2_j_teacher500h_node8_20261102/production/`。原媒体、标注和 AV/Gemma 缓存：`../p2_j_train500h_cache_20261101/`。权重：`../p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102/final_adapter/`。

以状态文件为准，不能把本说明当作 GPU 验证或全量完成证明。未启动训练。
