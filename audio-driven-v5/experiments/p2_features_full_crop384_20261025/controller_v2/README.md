# CPU 控制器待校验队列提速（2026-10-25）

同一全量特征流水线的控制器版本替换，不新建任务或定时器。现有 28 卡、56 个 GPU worker、节点监督器及 20 分钟巡检继续运行。

随着检查点增多，控制器一轮扫描约 7.4 秒。原实现同时只持有 8 个校验 future，约每轮最多交付 128 段校验，实际校验降至 16.9–18.8 段/秒，低于约 20.2 段/秒的特征抽取，积压增长到约 2,000 段。

当前副本与原 `scripts/manage_p2_full_features.py` 目前有三处代码差异：

1. `ROOT` 明确指向原工程，保证脚本移位后依然调用原模型代码、节点 worker 和状态页面路径。
2. 待处理作业数从固定 8 改为配置 `controller_pending_jobs`，当前 64。实际 `ProcessPoolExecutor(max_workers=8)` 不变，已有 CPU 校验进程数不增加；它们可在控制器扫描期间继续处理缓冲作业。
3. 用户停止定时巡检后，控制器遇到 `monitor/FINISHED.json` 跳过监控启动，防止守护重新拉起。`monitor/DISABLED.json` 保存用户指令、进程身份及验证记录。

`audit_job`、`final_index`、文件 SHA-256、contract identity、样本归属及完整数量／唯一性检查均逐字保持。主运行 `code_manifest.json` 的冻结源码没有修改，所有 GPU worker PID 保持原值。替换时只结束经 PID、命令、进程组核对属于本运行的旧 CPU 控制器／守护及校验子进程，由新版本接管同一把锁和完成记录；未完成校验可重试，已有缓存没有删除。

- `test_dispatch.py`：3 项通过，包括两轮调度维持 8 个进程、64 个在途作业无重复；代码仅上述三处改动、已停用监控不重启；有效缓存通过而损坏缓存拒绝。
- `code_manifest.json`、`deployment.json`：版本哈希、旧／新进程及启动记录。
- `baseline_status.json`、`config.before.json`：切换前状态与配置。
- `../monitor/optimization-1792960271.json`：本轮测量与运行验证。

当前控制器恢复入口（已有守护运行，正常无需重复执行）：

```bash
RUN=/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_features_full_crop384_20261025
/private/software/conda/envs/vllm/bin/python \
  "$RUN/controller_v2/manage_p2_full_features.py" \
  --run "$RUN" --role controller --watchdog
```

保持运行配置 `controller_pending_jobs=64`。修改该值需仅重载 CPU 控制器后生效。回退时按同样的进程身份和锁检查恢复原入口，或降低此配置；无需重启 GPU 抽取。节点与监控本身无改动，现有 1,200 秒报告周期不变。
