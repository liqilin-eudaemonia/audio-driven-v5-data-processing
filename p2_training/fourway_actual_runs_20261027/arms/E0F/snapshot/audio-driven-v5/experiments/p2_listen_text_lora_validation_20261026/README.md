# P2 text-attention LoRA 验证记录

2026-10-26。[实现与运行](../../related/P2_TextAttention_LoRA实现与验证.md)。本轮仅完成代码及 CPU/真实缓存验证，无正式训练或新 GPU S0。

- [summary.json](summary.json)：最终状态与验证边界。
- [baseline_verified.json](baseline_verified.json)：实施前核验 GitHub c72c10a 的备份和当前源码。
- [implementation_plan.md](implementation_plan.md)：已授权实施范围。
- [unit_tests.json](unit_tests.json)：136 项 CPU 测试通过；代码哈希一并记录。
- [cpu_ddp_lora.json](cpu_ddp_lora.json)：含 LoRA 的单/双进程 E0/E1 参数差为 0。
- [cpu_ddp_frozen.json](cpu_ddp_frozen.json)：无 LoRA 模式回归。
- [cpu_audit_lora.json](cpu_audit_lora.json)：事件梯度、AV 到 LoRA 梯度和精确恢复。
- [real_cache_validation.json](real_cache_validation.json)：50 条真实缓存、无 teacher、真实 LTX 架构的 meta 检查。
- [validate_real_caches.py](validate_real_caches.py)：上述检查脚本，不加载预训练张量。
- [runtime/config.json](runtime/config.json)：50 段 debug 缓存对应的新模式配置。
- [gpu_availability.csv](gpu_availability.csv)：指定机器全部 GPU 被使用的检查记录。
- [cache_and_backup_protection.json](cache_and_backup_protection.json)：冻结缓存源、v4 源码与全量抽取配置核验。
- [final_code_manifest.json](final_code_manifest.json)：最终代码和配置哈希。

CPU 复现：

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$P2_PY" scripts/check_p2_listen_code.py --output /tmp/p2_text_lora_unit_tests.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$P2_PY" scripts/p2_listen.py audit --config configs/p2_listen_av_event_text_lora_v1.json --suite cpu --output /tmp/p2_text_lora_cpu_audit.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$P2_PY" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/check_p2_listen_cpu_ddp.py --config configs/p2_listen_av_event_text_lora_v1.json --output /tmp/p2_text_lora_cpu_ddp.json
```

LoRA CPU 测试使用当前环境的原生 `ltx_core.model.transformer.attention.Attention`，随机小模型，无预训练权重。GPU 空闲后仍需用 runtime 配置跑两组 S0、GPU 恢复/独立生成和分布式检查；debug 数据不能进入正式效果训练。
