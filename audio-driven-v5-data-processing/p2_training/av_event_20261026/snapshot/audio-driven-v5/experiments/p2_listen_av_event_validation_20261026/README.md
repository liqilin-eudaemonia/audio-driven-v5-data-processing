# P2 AV / AV＋Event 简化验证

2026-10-26。代码实现和 CPU/真实缓存检查完成；本轮未重新执行完整 LTX GPU S0，未启动正式效果训练。详见 [实现与运行说明](../../related/P2_AV与事件监督简化实现.md)。

- [summary.json](summary.json)：最终状态、已完成检查与 GPU 验证边界。
- [unit_tests.json](unit_tests.json)：119 项 CPU 测试，0 失败/错误/跳过。
- [cpu_audit.json](cpu_audit.json)：无 feature 的 E0/E1 梯度与精确恢复。
- [cpu_ddp_av_only.json](cpu_ddp_av_only.json)：无 feature，两进程 E0/E1 对照均为零参数差。
- [cpu_ddp_with_features.json](cpu_ddp_with_features.json)：完整模式回归，两组同样为零参数差。
- [real_cache_validation.json](real_cache_validation.json)：没有 teacher 目录时加载 50 段真实 input/vae/text/events，通过身份/哈希/有限值检查。
- [validate_real_caches.py](validate_real_caches.py)：可重复执行的真实缓存验证。
- [runtime/config.json](runtime/config.json)：50 段 debug 弱背景数据的无 feature 配置；cache 四部分及 manifests/labels 使用既有结果的软链接，无 teacher 链接。
- [backup_verified.json](backup_verified.json)：改动前源码备份核验。
- [final_code_manifest.json](final_code_manifest.json)：最终源码与配置身份。

CPU 复现：

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PY" scripts/check_p2_listen_code.py --output experiments/p2_listen_av_event_validation_20261026/unit_tests.json
CUDA_VISIBLE_DEVICES='' "$P2_PY" scripts/p2_listen.py audit --config configs/p2_listen_av_event_v1.json --suite cpu --output experiments/p2_listen_av_event_validation_20261026/cpu_audit.json
CUDA_VISIBLE_DEVICES='' "$P2_PY" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/check_p2_listen_cpu_ddp.py --config configs/p2_listen_av_event_v1.json --output experiments/p2_listen_av_event_validation_20261026/cpu_ddp_av_only.json
CUDA_VISIBLE_DEVICES='' "$P2_PY" experiments/p2_listen_av_event_validation_20261026/validate_real_caches.py
```

GPU 空闲后可用 `runtime/config.json` 做新的 E0/E1 S0；这些标签保留 debug 身份，不能通过删除 `--s0` 启动正式训练。使用新的 run seed 并按实际空卡设置 `CUDA_VISIBLE_DEVICES`。
