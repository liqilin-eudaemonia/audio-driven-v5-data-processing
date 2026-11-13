# 新筛选规划LoRA：三机24卡供数加速（解码隔离版）

使用 node17 `172.29.15.91`、node12 `172.29.252.37`、node4 `172.29.225.51`，各8卡。node8释放。

## 恢复与训练约定

从原四机完整 checkpoint-500 恢复模型、优化器、学习率进度与保留节点的RNG。第一次三机尝试在575步因CPU解码worker停滞停止，未落盘的501–575重算；旧目录及停滞记录保留。

- train 690,337条／1,155.44h；dev不参与训练且不运行评估。
- 全局batch256，每rank10或11条；全局样本批次与原32卡一致。1轮总2,697步，每500步保存，末尾另存final_adapter。
- 学习率、LoRA参数、目标文本、speaker音频、listener最终裁框与首帧选择规则保持不变。改变DDP拓扑后不承诺逐位相同的后续训练轨迹。
- 首帧旧PNG逐条核对完整来源、crop和像素hash后复用；其余每worker独立持久CPU进程AUTO4解码，30秒超时重启并回退原读取器，再失败明确报错、不跳样本。
- 每卡4个DataLoader worker，prefetch4；解码子进程不占GPU，临时RGB传回后删除，不新建全量媒体文件。

## 依据与状态

- `input_equivalence50.json`：50条真实媒体，首帧RGB／PTS、音频、文本、processor张量完全一致；本轮原读取81.23秒、隔离读取10.83秒。仅为输入端测试，不代表训练总吞吐。
- `tests.log`：29项回归通过，含真实H264近邻帧选择及模拟卡死重启后串行回退。
- 原全局分组验证：`../p2_planning_sft_filtered_3b_32gpu_20261102/performance/batch_partition24.json`，全部2,697组一致。
- `performance_load.json`：启动后adapter／optimizer／scheduler恢复审计。
- `rng_restore/`：24个rank的RNG恢复审计。
- `step_metrics.jsonl`：实际步进与吞吐，以此判断运行，不以启动命令成功为依据。
- `decoder_logs/`：超时／重试的源数据记录；`source_snapshot/`为本次训练代码快照。

## 启动与续跑

初次启动（当前已启动，勿重复）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_planning_cluster.py --config configs/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102.json --cluster configs/p2_planning_cluster_filtered_24gpu_20261102.json --launch
```

若本次运行中断，使用同一代码／配置，在上述命令增加 `--resume <本目录内最近完整的checkpoint-N>`；须先确认该检查点的模型、optimizer、scheduler、trainer_state及24份rng_state均完整。没有新完整checkpoint时仍从原500恢复，使用新的输出目录并重新固定迁移证明，勿覆盖run.json。

来源与切换说明：[交接文档](../../related/P2_Qwen规划LoRA供数加速与无损接续_20261102.md)。

## 实际运行核对

2026-11-02T07:38:16.006007+00:00：已从500恢复到686步，越过原575停滞点，24个rank恢复和梯度审计通过。最近100步（587–686）平均1.449秒/步，约176.7样本/秒；原32卡401–500窗口约4.486秒/步。不同样本窗口，仅为当前吞吐观察，非严格AB。解码超时／错误日志数量0。这表示当前训练推进正常，不表示全程训练已经完成。[审计](switch_verified.json)。
