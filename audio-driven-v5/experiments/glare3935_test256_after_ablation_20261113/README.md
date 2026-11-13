# GLARE step3935：接续当前消融任务后的Seamless固定256条测评

**最新调度（2026-11-13）：用户要求先不使用node8，利用node12／node11空卡。** 当前消融768条视频已全部生成，仅剩评分收尾；本轮允许在其他空卡提前启动GLARE，自动避开消融评分仍占用的GPU。后者结束后使用两机全部16卡。`run_two_nodes.py`使24个逻辑分片可由任意空闲授权GPU领取，不再将其中8片固定到node8等待。样本、权重、seed137、NFE10、CFG2、解码batch4和评分公式不变。`ALLOW_TAIL_OVERLAP.json`记录本次授权范围，`TWO_NODE_RESUMED.json`记录资源切换。下方“三机24卡／全部评分后才接续”为初始安排，现由本段替代；恢复仍使用handoff.py，它会启动run_two_nodes.py。


2026-11-13用户授权：当前12组消融（每组64条）全部推理、评分完成并释放控制器锁后，在node8/node12/node11各GPU0–7自动接续本次评测。

- 权重：早期batch8训练 `../glare_500h_training_node11_gpu7_20261111/latest.pt`，内部step=3935。本目录 `checkpoint_step3935.pt` 通过硬链接保留当前inode，防止原latest被原子替换后误用新权重；实际SHA见 `CHECKPOINT.json`。
- 范围：原固定Seamless256条，seed137；与Qwen7500＋LTX6800及GLARE10618／19050同一清单、首帧、speaker音频、时间范围和GT。256条源媒体及官方韵律缓存复用，全部生成重新执行。
- 原生流程：复用既有GLARE评测入口、ResidentGLARE／ODE／LIA；NFE10、speaker CFG2、解码batch4。每卡最多一个worker，生成22GiB／评分24GiB上限，启动至少48GiB空闲；24个独立分片生成后各自接评分，batch64、像素线程4及GT缓存保持。
- 最终全256条特征合并FID/FVD，不平均分片分数；GLARE无listener音频预测，音频指标N/A。
- 3935是另一条早期batch8训练线，不作为batch256训练过程的中间检查点陈述。训练参数和资产身份完整记录在 `CHECKPOINT.json`。

## 状态及结果

`QUEUE_STATE.json`：等待／接续状态。当前测评只有 `step3935/COMPLETE.json` 和总 `COMPLETE.json` 出现后才完成。

`step3935/status.json`：生成数、评分分片及活跃worker。

`step3935/RESULTS.json` / `RESULTS.md`：最终结果与有效覆盖。

完成后自动刷新[固定256条seed137结果表](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/#seamless256-seed137)，不混入seed238或双seed均值。

## 恢复

已挂载的接续进程见 `QUEUE_CONTROLLER.json`，日志 `handoff.log`。`handoff.py` 使用队列锁防重复启动，并核对上游12组、768条成功收据及控制器锁释放；上游暂停、未完成或失败时不抢占GPU。当前消融任务原代码不改。

重新挂载前确认旧接续进程已退出；在仓库根目录执行：

```bash
nohup /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u \
  experiments/glare3935_test256_after_ablation_20261113/handoff.py \
  >> experiments/glare3935_test256_after_ablation_20261113/handoff.log 2>&1 &
```

本目录 `STOP` 会阻止尚未开始的接续，或让已开始的GLARE控制器停止；只关闭本次启动的worker，其他任务不动。移除STOP应依据用户恢复指令。`QUEUE_FAILURE.json` 或 `controller.log` 记录故障。已完成视频／评分分片保留，恢复会跳过；至多3次执行重试。

CPU核验涵盖：与10618已验收模型的参数名／shape／冻结资产一致、3935权重有限值及SHA、输入精确一致、24分片互斥覆盖256条、未完成/暂停/锁未释放时禁止接续、完成数量错误拒绝。新3935权重的实际GPU加载及推理验证将在轮到本任务时执行，不抢占当前测评。
