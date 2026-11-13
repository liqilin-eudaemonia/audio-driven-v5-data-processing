**已于6026步按用户要求完整保存停训，后续迁移到[三机24卡](../p2_ltx_clip_train24_reserve_node11_20261107/README.md)，node11整机预留推理测评。以下是历史31卡运行记录。**

# 四机31卡接续：留出推理卡，按5500步切换保存频率

从原32卡3982步完整模型、Adam和RNG接续，当前固定all全重计算。global batch31，每个rank每次更新1条；原模型、条件、AV/整段Event/双路Feature目标与优化器参数保持。31个rank的模型/Adam及对应RNG逐值迁移通过，原32份RNG保留在父检查点。

- node12、node8、node4各用GPU0–7；node11仅用GPU0–6。**node11（172.29.189.119，SSH22／外部标识20476）的物理GPU7整卡留给推理**，不在训练进程CUDA_VISIBLE_DEVICES中。
- 5500步前按1000步周期保存：4000、5000；从5500步起按500步周期保存：5500、6000、6500……9000。最终9041步另存完整检查点。保存条件直接由训练循环执行，5500步无需重新启动。
- 旧32卡消耗前缀保持。剩余156800个原(sample index,noise ordinal)配对完整保留，重新按时长分组为31条/批（每窗496条），仅末批追加29条填充；预算调整为9041步以覆盖全部剩余数据。分组改变，样本与各自噪声编号不变。未重新打标或重新提取缓存。
- 原9小时goal已由用户暂停，旧守护STOP保留，不会按旧32卡/每300步配置自动拉起。训练进程和只读性能监控继续运行。

[实际状态](status.json) · [当前配置](config.json) · [迁移逐值验收](MIGRATION.json) · [实际更新验收](RUNNING_VERIFIED.json) · [31rank进程/推理卡隔离](LIVE_PROCESSES.json) · [时长分组效果](DURATION_REBATCH_VALIDATION.json) · [实时性能](PERFORMANCE_STATUS.json)。

CPU验证：48项相关测试通过；原训练数学计算除保存条件和必要的接续入口校验外，按文本还原后与父版SHA一致。旧版3个修改文件归档在source_before/；unadopted_exact_order/仅为未启动训练的临时迁移副本，不用于恢复。当前恢复须使用本目录配置和scripts/p2_ltx_resume31.py入口，不直接套回旧32卡配置。

## TensorBoard

入口：[node12:6006](http://172.29.252.37:6006)。服务已切换到本目录tensorboard/，保留原训练曲线并继续写入31卡接续数据。服务不占GPU；身份和启动命令见[TENSORBOARD_SERVER.json](TENSORBOARD_SERVER.json)。外部访问使用原node12 SSH参数，追加-L 16006:127.0.0.1:6006，再打开http://127.0.0.1:16006。

本目录STOP会让训练在当前更新完成后完整保存退出；不终止其他项目。controller身份见CONTROLLER.json，四节点启动日志见launch/，逐rank日志见rank_logs/。
