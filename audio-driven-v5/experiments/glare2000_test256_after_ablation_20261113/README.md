# GLARE早期batch8 step2000：同Seamless256、seed137

用户要求补测step2000。采用 `experiments/glare_500h_training_node11_gpu7_20261111/best.pt`，读取内部step确认2000，并固定硬链接 `checkpoint_step2000.pt` 及SHA。该权重与step3935属于早期batch8训练线，不能视为后期batch256训练线的中间权重。

复用已验收的GLARE3935评测入口和官方模型推理实现，仅替换权重、独立输出目录、两机分配与接续条件。相同256条／GT、seed137、NFE10、speaker CFG2、解码batch4；输入及官方韵律缓存复用，256条视频全部重新生成。评分沿用GPU检脸／表情、batch64、像素4线程；24个逻辑评分分片汇总完整256条特征计算FID/FVD，音频指标N/A。

## 队列与结果

先等待 `../p2_ablation6_test256_qwen7500_16gpu_20261113` 的6×256推理评分全部完成并释放控制器锁，再在node12、node11各GPU0–7启动，共16卡。node8不参与，不停止当前消融任务。所有逻辑分片可由任意空闲授权卡领取；临时失败最多重试3次，完成产物断点复用。

- `QUEUE_STATE.json`：等待／执行／完成状态；`QUEUE_CONTROLLER.json`：接续进程。
- `CHECKPOINT.json`：实际step、来源、SHA、模型结构核验。
- `step2000/status.json`：生成数及评分分片进度。
- `step2000/seamless/generated/`：生成视频；`step2000/RESULTS.json`：最终原始指标。
- `step2000/COMPLETE.json`、根目录 `COMPLETE.json`：评测完成；`QUEUE_COMPLETE.json`：页面更新完成。
- `queue.log`、`controller.log`、`step2000/logs/`：运行日志；失败记录 `QUEUE_FAILURE.json`、`FAILURE_two_nodes.json`。

完成后自动更新 [固定256条结果页](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/#seamless256-seed137)，第一名加粗、第二名下划线；检测有效覆盖按实际结果保留。

## 恢复

在node12用 `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python` 运行本目录 `handoff.py`，会先核对冻结身份与前序完成情况。不要重复启动；`queue.lock`／`controller.lock`防止重复运行。创建本目录 `STOP` 停止分发，恢复前删除。保持权重、协议和完成收据不变；恢复跳过已完成的生成与评分。
