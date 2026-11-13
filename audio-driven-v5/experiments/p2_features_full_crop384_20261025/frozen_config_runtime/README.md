# 冻结配置模块续跑恢复（2026-10-26）

共享源码 `src/p2_listen/config.py` 在 03:29 UTC 增加了训练侧 feature-loss 开关。整文件哈希变化触发抽取 worker 的严格版本保护，最终停在 664,598 段；已有结果及原始媒体未损坏。

本次不覆盖共享源码、不简单接受新哈希，也不关闭校验。运行配置 `python` 指向本目录 `frozen_python`，其 shebang 仍是原抽取 Python 环境；它先将本轮保存的 `code/audio-driven-v5/src/p2_listen/config.py` 注册为 `p2_listen.config`，核对原 SHA-256，再执行原 `scripts/run_p2_full_features.py`。原工程的 contracts ROOT、features 文件位置、模型配置、缓存身份保持不变。此启动器仅允许原抽取脚本和本目录验证脚本，不用于训练。

`code_manifest.json` 中原共享 config.py 的校验目标替换为**实际导入的冻结快照**，期望内容哈希不变，同时加入启动器本身的哈希；其余源码校验项全部保留。其他共享文件如有改动仍会触发保护，不能认为全部源码都已隔离。

- `validate_bootstrap.py`：核对冻结模块加载及原工程 ROOT，比较 3 个已完成样本各 input/teacher/vae/events 共 12 项与共享 text，共 13 项缓存身份；全部与原完成标记严格一致。
- `validation.json`：验证结果及保留的当前共享源码哈希。
- `code_manifest.before.json`、`config.before.json`、`baseline_status.json`：变更前记录。
- `restart-<node>.json`：按命令、运行目录、worker 名称、进程组及授权 GPU 核对后，仅重启本任务的节点监督器和抽取 worker。既有 watchdog 接续启动，node8 GPU0–3 其他训练不动。
- `deployment.json`：恢复状态、版本与进度。既有作业错误只在实际重试成功后标为已解决。

共享源码保留新增训练功能。新的 worker 继续写原运行目录和缓存，从已有完成标记恢复；不重新打标，不改标签或音视频内容。每 20 分钟巡检／汇报仍为用户停用状态，没有恢复。

重跑 CPU 身份验证：

```bash
RUN=/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_features_full_crop384_20261025
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$RUN/frozen_config_runtime/frozen_python" \
  "$RUN/frozen_config_runtime/validate_bootstrap.py"
```
