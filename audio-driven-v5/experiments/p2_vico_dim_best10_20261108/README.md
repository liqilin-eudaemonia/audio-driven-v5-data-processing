# ViCo / DIM：固定十次采样与GT择优补测

**执行入口说明：** [EVAL_ENTRY.md](EVAL_ENTRY.md)汇总绝对路径、已完成的单次结果、只读校验、node11 GPU7 恢复命令及完成产物；同时明确用户指定 Var 的十次后处理尚未自动接入。本轮只交付说明，队列保持暂停。

用户要求在无法联系作者的情况下自行推断配置并继续推进。选择DIM发布tester明确存在的10次采样规则作为**补充实验**，沿用ViCo此前最后更新的联合8000＋SFT2752模型；不按此次新分数选择模型，不覆盖base-Qwen或SFT2752原单次结果。

**当前状态更新：已按用户要求暂停，为31卡训练保留资源。** `status.json`为`paused_by_user_for_31gpu_training`；本次Var公式修改未恢复该队列。此前的容量轮询和进程1278570为启动历史，不代表现在仍活跃。[实际状态](status.json) · [历史启动进程](LAUNCH.json) · [原环境CPU预检](CPU_ACCEPTANCE.json)。下一次评分需采用[已指定的逐系数Var口径](../p2_vico_dim_var_per_coefficient_20261108_v2/README.md)，要求记录见[REQUESTED_VARIANCE_READOUT.json](REQUESTED_VARIANCE_READOUT.json)。

## 已固定的配置

- ViCo正式test43，独立listener UUID，不含OOD。
- 同一联合8000权重、SFT2752条件、speaker Whisper、listener首帧和原生512／25→评分256／30流程；Euler30／原CFG等全部沿封存协议。只变化LTX随机种子，不重新生成Qwen描述或H。
- 固定seed137–146。已有seed137的43条媒体及EMOCA特征直接复用；补9×43＝387条，最终430个候选。
- 每条在所有十候选共同有效的原始帧上计算完整56维GT Fréchet，选择距离最小的一条；FD、P-FD、MSE、SID、Var、rPCC全部使用这个同一候选，不能表情／姿态分别挑。
- 单次137和十次择优另列，使用相同的十候选共同有效帧作配对比较；保留缺帧数及每条所选种子／候选距离。
- 该规则使用GT，是oracle评测，不能描述成部署时可以这样选择。发布代码含此路径，是否完全对应已发表主表仍未被证明；GT统计、RGB反提取等差异继续披露。

[固定协议及源码指纹](protocol.json) · [387条准备记录](PREPARED.json) · [队列／采样／提取／择优入口](../../scripts/p2_vico_dim_best10.py)。原始来源：[封存SFT2752生成](../p2_vico43_combined8000_sft2752_20261107/README.md)、[原EMOCA特征](../p2_vico_literature_protocol_20261108/README.md)、[已完成的配置敏感性评分](../p2_vico_dim_inferred_protocol_20261108/sensitivity_v1/RESULTS.md)。

## 自动接续与资源

只用node11物理GPU7及已核对UUID。每个种子按latent→decode→EMOCA顺序执行，GPU模型分阶段退出后再进入下一阶段；不存在跨种子并行抢占。首档采样要求至少88,000MiB空闲、进程上限40%；只有实际OOM时才保留已提交输出并等待110,000MiB后以60%上限恢复。解码要求60,000MiB，EMOCA要求32,768MiB，均保留已有任务。

本轮最后实测空闲38,718MiB，低于采样门槛，因此队列等待，每30秒更新一次。等待状态不是生成已完成，也不是模型加载失败。不停止其他任务，不使用0–6卡，不需要作者回复或新的人工确认。

每次执行前核验冻结源码与上游协议，变更会明确失败而不静默切换。保留各seed的阶段日志、退出码、latent／媒体提交记录和提取回执。除latent明确OOM的一次增额恢复外，不自动重复失败计算；异常见`status.json`和对应日志。

4项新增测试已通过：完整56维选择同一候选、相同距离保留首候选、缺失／错位候选拒绝、GPU UUID与余量准入。原环境预检验证真实SFT adapter和同一43条缓存，CUDA未初始化；这不冒充GPU实跑验收。

## 后续产物

所有387条完成后自动输出`RESULTS.json`、`RESULTS.md`、`COMPLETE.json`，含原单次与best10的同覆盖比较及每条选取证据。当前这些最终成绩尚未产生，不能引用为已完成。

只有确认原后台进程已退出后，才用以下入口恢复；`controller.lock`会阻止同目录重复协调器：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python scripts/p2_vico_dim_best10.py controller
```

该命令应运行在node11；worker自行绑定物理GPU7。历史封存目录只读，新输出全部进入本目录`draws/<seed>/`。
