# P2 50 段真实 GPU S0：2026-10-25

**验收通过，仅用于代码测试。** 机器 `172.29.104.220`，使用当时空闲的 H200 GPU 4/5/6，任务完成后已释放。输入为 `../listen_debug50_fullraw_20261025/` 的 50 段、213.0865 秒数据；debug 标签无摇头正例、无已知负例，unknown 未修改。

结论与限制见 [完整验收记录](../../related/P2_50段真实GPU调试与训练就绪记录.md)、[summary.json](summary.json)。

## 产物

- [config.json](config.json)：全部 50 段，512×512，global batch 8；五类缓存位于 `cache/`，约 516.6 MiB。
- [cache_validation.json](cache_validation.json)：input/teacher/vae/events 各 50 份、text 1 份，身份/哈希/有限值/逐类统计。
- [runs/s0_20261026](runs/s0_20261026/)：E0/E1 各 7 步，完整 50 段样本覆盖；E1 第 5 步暂停后恢复至第 7 步。
- [ddp/runs/s0_20261026/E1](ddp/runs/s0_20261026/E1/)：双卡 NCCL 7 步；与单卡对照见 [comparison.json](ddp/comparison.json)。DDP 的 cache/manifests/labels 通过只读使用的软链接复用主目录。
- [smoke](smoke/)：单样本 E1 5 步、每步 checkpoint、源输入隔离生成、恢复重放。
- [短片段输出](smoke/generated_source_only/generated.mp4)、[最长片段输出](generated_longest/generated.mp4)：仅验证生成链路；不作为质量结果。
- [GPU 恢复重放](smoke/replay_validated.json)：状态精确恢复，loss/RNG 一致，GPU 更新按明确容差验收。`replay.json`、`replay_attempt2.json` 为已诊断的早期检查记录，不是最终失败项。
- [unit_tests.json](unit_tests.json)：72 项 CPU 测试通过；[formal_guard_validation.json](formal_guard_validation.json)：debug 标签被正式训练入口拒绝。
- [input_manifest.json](input_manifest.json)、[input_integrity_final.json](input_integrity_final.json)：输入清单未改动；`before/` 保留修改前代码/文档；[final_code_manifest.json](final_code_manifest.json) 记录最终文件身份。

## 已执行的主要命令

工作目录为 `/private/liqi_workspace/project_code/audio-driven-v5`，Python 为 `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。下面省略 SSH 外层；`P2_PYTHON`、`P2_CONFIG` 只是便于阅读的变量。已有输出受严格身份保护，复现新实验需要新的 output；不要重复向现有 run 追加。

```bash
P2_PYTHON=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
P2_CONFIG=experiments/p2_listen_gpu_debug50_20261025/config.json

"$P2_PYTHON" scripts/p2_listen.py prepare --config "$P2_CONFIG" \
  --observations experiments/listen_debug50_fullraw_20261025/observations.json \
  --targets experiments/listen_debug50_fullraw_20261025/targets.debug.json
"$P2_PYTHON" scripts/p2_listen.py import-labels --config "$P2_CONFIG" \
  --labels experiments/listen_debug50_fullraw_20261025/labels.debug-positive.jsonl

# 两个独立进程分片缓存；共享固定文本由 shard 0 负责。
CUDA_VISIBLE_DEVICES=5 "$P2_PYTHON" scripts/p2_listen.py cache \
  --config "$P2_CONFIG" --part all --split train --shards 2 --shard-index 0
CUDA_VISIBLE_DEVICES=6 "$P2_PYTHON" scripts/p2_listen.py cache \
  --config "$P2_CONFIG" --part all --split train --shards 2 --shard-index 1

CUDA_VISIBLE_DEVICES=5 "$P2_PYTHON" scripts/p2_listen.py train \
  --config "$P2_CONFIG" --arm E0 --train-seed 20261026 --s0 --steps 7 --debug-positive-events
CUDA_VISIBLE_DEVICES=6 "$P2_PYTHON" scripts/p2_listen.py train \
  --config "$P2_CONFIG" --arm E1 --train-seed 20261026 --s0 --steps 7 \
  --stop-after 5 --debug-positive-events
CUDA_VISIBLE_DEVICES=6 "$P2_PYTHON" scripts/p2_listen.py train \
  --config "$P2_CONFIG" --arm E1 --train-seed 20261026 --s0 --steps 7 --debug-positive-events \
  --resume experiments/p2_listen_gpu_debug50_20261025/runs/s0_20261026/E1/step_000005.pt

CUDA_VISIBLE_DEVICES=4,5 "$P2_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/train_p2_listen_distributed.py \
  --config experiments/p2_listen_gpu_debug50_20261025/ddp/config.json \
  --arm E1 --train-seed 20261026 --s0 --steps 7 --debug-positive-events
```

`--debug-positive-events` 仅供这批弱正例检查。正式数据不能使用它绕过类别支持或准入要求；正式实验建立新的 output、正式标签和 split 后，按 [运行说明](../../related/P2_倾听事件监督实现与运行.md) 接入。
