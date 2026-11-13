# GLARE 500h 正式训练：node11 GPU7

**已停止并保存（2026-11-11）：** 用户要求对齐论文lr／有效batch／FP16，旧训练在3935步正常保存退出。当前新运行见`../glare_paper_settings_node11_gpu7_20261111/README.md`；下方运行中记录为历史。


2026-11-11 用户明确授权启动。仅物理GPU7，保留同卡原vLLM服务，不动其他任务。

首轮35133次优化器更新、每次8pair（281064次样本读取，覆盖281060条train并在边界重复4条），lr/weight_decay=1e-5，官方10%warmup及余弦调度、梯度裁剪1。每2000步保存latest.pt，每2000步对固定64条dev子集评估并保存best.pt。原3158条dev不参与参数训练；此轮不是官方100epoch完整训练预算。

入口：`scripts/train_glare_cached.py`、`scripts/run_glare_cached_training.sh`。准确启动命令见launch.json；冻结运行配置和模型源码/资产指纹见protocol.json。

- 日志：train.log；逐步指标：metrics.jsonl。
- 最新恢复权重：latest.pt；最佳验证权重：best.pt（首次验证后生成）。
- 已验证前135步loss/梯度有限，100步检查点原子写入成功（约1.94GB）。训练正在后台继续。

恢复前先确认原PID及子进程已经退出，避免重复训练。使用launch.json的原命令追加`--resume 本目录/latest.pt`，仍设置CUDA_VISIBLE_DEVICES=7、NPROC_PER_NODE=1、MASTER_PORT=29679。不改变steps、world、accumulate或已冻结源码；否则身份检查拒绝恢复。未添加无限自动重启或新的持续goal。

用户要求后已调整保存／验证间隔为1000步，并从latest.pt冷恢复；实际新进程和命令见launch.json，原命令存于launch_before_save1000.json。

最新覆盖：保存及验证均改为每2000步，见launch.json；此前1000步设置为历史记录。

执行优化已启用：入口为`scripts/train_glare_cached_fast.py`与`run_glare_cached_training_fast.sh`，准确命令见launch.json。原2000步检查点保留为original_step2000.pt，协议为protocol_original.json；迁移核验见execution_migration.json。位置编码/mask有界缓存＋loss批量同步，保持原模型、顺序、batch及2000步保存。受控短测耗时降低27.7%；上线实测见`../glare_training_speed_20261111/PRODUCTION_RESULT.json`。恢复必须使用新版入口，不能误回原入口。
