# ViCo新增基线：运行与结果

[43条同步检查入口](http://localhost:8876/listen_benchmark_review/vico_more_20261107/) · [结果及解释](RESULTS.md) · [比较JSON](COMPARISON.json) · [论文表格PDF](paper_tables/preview.pdf) · [实时状态](status.json)

截止：2026-11-08 北京时间12:00（04:00 UTC），用户已确认。ListenFormer原定00:00 UTC冻结；RLHG按实测后处理耗时的3倍（至少90分钟）预留生成、评分与验收时间，完整500轮预算另加15%余量，详见rlhg/RUN_BUDGET.json。
用户已确认暂时仅用node11 GPU7，优先DualTalk＋ListenFormer。现有31卡及AV＋Feature接续队列不改动；先完成ListenFormer，RLHG只在剩余时间足够跑满500轮时再启动。OVI、GLARE不做本轮。

DualTalk：官方完整权重严格加载、932个状态键逐值核验、官方demo核心输出误差0；43/43条已生成。原生25fps动作→30fps时间戳映射，self音轨置零，额外speaker动作输入。仅动作迁移代理；原实现称neck、DECA称global head，对应几何尚未验收，所以不进入共同RGB／头部优劣主张。

RLHG：完整500epoch、有效batch16。ListenFormer：完整200epoch、batch128、100epoch阶段切换。351train/46validation；test43不参与统计或选权重。完整候选训练预算不足时保留未完成原因，不拿短训充正式结果。

两方法共用BFM73／audio45缓存；音频与官方函数真实样本逐值一致，人脸对齐像素一致。训练器日志同步／padding掩码适配已核验前向、loss与全部梯度。源码与兼容层分开；PIRender EMA严格加载且真实训练片段前向通过。

| 方法 | 当前状态 |
|---|---|
| rlhg | {'reason': 'Both measured full500epoch ETAs plus15% margin cross measured delivery cutoff', 'eager_hours_with_margin': 12.605991793274876, 'compiled_hours_with_margin': 12.01894643008709, 'model_freeze_unix': 1794104787.8914511, 'formal_result': False} |
| listenformer | complete |

[批准设计](DESIGN.md) · [源代码版本](REPOSITORIES.json) · [协议](protocol.json) · [控制器状态](CONTROLLER_STATE.json)
[持续执行入口](../../compare/deployment/vico_more_controller.py) · [训练／推理](../../compare/deployment/vico_more_train.py) · [缓存](features/protocol.json)

旧SFT2752、DyStream、Avatar Forcing、L2L及全部训练输入保持原样。视觉相似度、动作距离和动作幅度共同报告；尚未收集反应适当性人工评分。

[完整逐条指标CSV](paper_tables/all_motion_per_clip.csv) · [配对区间CSV](paper_tables/paired_sft2752_minus_new.csv) · [条件与训练口径](paper_tables/conditions.csv) · [全部方法完成范围](METHOD_COVERAGE.json)
[训练／验证与test来源时间隔离](SPLIT_ISOLATION_AUDIT.json) · [PCH可行性与未列正式行的原因](PCH_FEASIBILITY.json)

恢复：在项目根目录运行 `python compare/deployment/vico_more_controller.py`。已有控制器持锁时不重复启动；已经封存完成时仅校验文件，不重新跑模型。各阶段日志与实际退出状态保存在 `jobs/<stage>/`。
