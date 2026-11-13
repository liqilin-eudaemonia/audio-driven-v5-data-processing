# Avatar Forcing：显式固定运行状态的独立入口

已实现并实际验证。当前只生成了用于定位问题的1条异常片段，**没有启动新的43条全量生成或评分，尚未复现Table4**。完整调查见[差异定位报告](../p2_avatarforcing_table4_diagnosis_20261110/README.md)。

修正位置：[生成入口](../../scripts/p2_avatarforcing_table4_generate_stable.py)。人脸检测器初始化和预处理结束后，每次模型前向之前固定`cudnn.benchmark=False`、`cudnn.deterministic=True`、两项TF32=False；逐条记录实际状态与导出min/max。保留官方原网络、权重、CFG2／1、NFE10、seed137和官方整段min/max导出公式，仅隔离运行状态问题，不同时改变多个生成条件。

已在node11物理GPU7上验证`UCA1A5GqCdQ_001213_001223.1`，250帧。修正后视频与“原生成脚本只在前向前固定benchmark=False”的严格干预视频SHA完全相同。见[实际修复验证](FIX_VALIDATION.json) · [逐条回执](generated/UCA1A5GqCdQ_001213_001223.1/complete.json) · [冻结协议](protocol.json)。

已预处理输入只读复用上一版的完整缓存；新视频写入本目录。上一版43条视频、动作、分数及代码均未改写。作者私有Table4的后端细节未披露；本版本是本地运行可控性修正，不代表已经确认原论文使用False。

在node11、项目根目录使用以下入口（以下命令列作复现实验记录，不表示已排队全量运行）：

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=experiments/p2_avatarforcing_table4_recheck_20261110/runtime_py \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/avatarforcing/bin/python scripts/p2_avatarforcing_table4_generate_stable.py outlier
```

本入口的`generate`动作会生成计划中的43条；后续完整评分需要独立冻结新协议与新产物路径。原来的SPECTRE／score控制器仍绑定上一版封存目录，不能直接重启它来声称给本版本评分。
