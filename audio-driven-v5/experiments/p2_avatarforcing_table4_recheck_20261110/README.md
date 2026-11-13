# Avatar Forcing Table4：官方权重端到端复现检查

**后续差异定位已完成：** 发现并验证了原入口／简化直调间的cuDNN状态与导出亮度差异，已提供独立修正版入口；同时确认作者发布示例与本地紧裁脸视野不同、三个评测来源的定义不能混用。原43条结果保持封存。请先读[完整定位报告](../p2_avatarforcing_table4_diagnosis_20261110/README.md)与[运行状态补充说明](RUNTIME_STATE_CLARIFICATION.json)；修正版只验证1条，不是新的43条评分。

**实际执行已完成。** 官方test43全部生成，129路SPECTRE动作与8项评分完成；8项中0项按论文两位小数一致；当前公开配置重建未复现Table4，不能宣称评测已严格对齐。

[逐项结果与论文差值](RESULTS.md) · [原始数值和敏感性结果](RESULTS.json) · [最终独立核验](FINAL_AUDIT.json) · [完成回执](COMPLETE.json)。

[实际状态](status.json) · [当前worker和日志](ACTIVE.json) · [控制器](LAUNCH.json) · [冻结协议](protocol.json) · [执行设计与推断边界](DESIGN.md) · [论文页与官方来源](SOURCES.json)。

## 为什么重新推理

11月7日的旧43条使用speaker音频＋listener首帧，speaker视觉条件置零。该适配适合共同输入比较，但不代表Avatar Forcing原版的完整输入。此轮恢复官方`encode_user_motion`，给入真实speaker视频与音频，保留旧视频和成绩不变。

使用官方`flow_transformer.pth`＋`motion_autoencoder.pth`及同一Wav2Vec权重，全部非Wav2Vec核心参数严格来源核验，Wav2Vec旧命名兼容后逐张量相等。固定seed137、CFG2／1、10 NFE、KV cache；完整波形不截到30秒。10 NFE是ODE推理步数，不是10次候选。

ViCo仅提供speaker音轨，本轮avatar音轨为全零，是明确记录的listening-only推断；Table4没有写明这一细节。原始数据是官方裁脸256／30，而作者描述自行裁脸512／25，因此沿官方预处理转换并披露这个来源差异。

## 本篇论文的指标配置

论文第14–15页D.5明确使用SPECTRE提取exp50／pose6，SID为K15／9，Var沿时间轴计算再沿特征轴平均。这与DIM的EMOCA、K40／20分支不同；Table4中带†的旧基线是引用DIM数字，不代表Avatar Forcing自己也用了完全相同的提取器和聚类数。

本轮共享已核验的Fréchet计算内核，新增SPECTRE配置分支。检查FD、rPCC、SID、Var各自Exp／Pose，共8个数值；P-FD／MSE不在Table4中。

| 指标 | 论文Avatar Forcing Exp | 论文Avatar Forcing Pose |
|---|---:|---:|
| FD | 16.64 | 0.05 |
| rPCC | 0.01 | 0.01 |
| SID | 3.12 | 3.00 |
| Var | 2.80 | 0.03 |

SID拟合／汇总、FD式17均值项等仍有文字歧义，保留预先声明的替代解释；不挑最接近论文的公式。SPECTRE采用官方checkpoint和编码函数、残差加回、五帧时间卷积和边界复制，全部645个编码状态张量严格加载。

另有一处论文内部一致性问题：按D.5所写pose K=9，标准以2为底的香农熵上限是log2(9)≈3.17，但Table4的INFP* Pose SID为3.20；Table1的GT SID 3.972也高于K15／9各自及其平均的上限。这不能判定实际代码用了什么，但说明附录不足以唯一确定所有表格的SID实现。本轮仍保留预先声明的K15／9，不据此反向调参，见[PAPER_CONSISTENCY_CHECK.json](PAPER_CONSISTENCY_CHECK.json)。Avatar Forcing自己的Table4目标3.12／3.00不超出这些上限。

## 已通过的检查与运行

- 最短1秒及最长约35秒实际生成试跑通过，产物留在正式43条中。
- SPECTRE官方编码与独立系数拆分完全一致；CPU整段／带上下文分块最大误差0。
- 4项科学计算测试通过：时间上下文完整覆盖、Fréchet均值项解析值、常量系数相关性未定义、SID K15／9的熵上限。
- 官方裁剪方法的命名空间／文件／512输出检查通过；已生成参考图、真实speaker和输出的[试跑画面对照](smoke_geometry_review.png)。

当前按生成→SPECTRE129路动作提取→CPU评分顺序自动执行；仅node11物理GPU7，预留至少16GiB，并分阶段检查空闲显存。新测试不恢复已暂停的DIM best10队列，不停止其他作业。

实际生成43／43已完成，共13,575帧、543秒，均为原生512／25，全部恢复真实speaker视觉且未输入listener未来画面。SPECTRE提取进行中，实时覆盖见`status.json`。GPU整段／时间上下文分块最大误差约9.54e-7，独立系数拆分误差0，见`SPECTRE_GPU_PARITY.json`。

批量人脸检测和仅合批FAN的加速候选均未通过数值一致性检查（关键点最大偏差7像素），正式提取继续使用原先逐帧路径；未改模型、未抽帧、未修改冻结协议，见`LANDMARK_BATCH_BENCHMARK_v1_REJECTED.json`及`LANDMARK_BATCH_BENCHMARK.json`。这些候选没有写入正式特征。

CPU收尾程序已接好：控制器完成后自动核验43条完整解码、129路特征／时间戳／校验和、独立方差与相关系数重算，保存`FINAL_AUDIT.json`，再更新本页及根README和evaluation/README中的索引。收尾不占GPU，状态见`POSTPROCESS_ACTIVE.json`，不需人工再次启动。

生成环境原先不含face-alignment，本次仅安装到本目录`runtime_py`，未修改旧环境。首次试跑缺依赖和随后裁剪适配漏传os的失败日志保留；修正前没有有效视频计入正式结果，见`revisions/preprocess_namespace/`。

## 最终如何判断

全部43条完成后，报告8个实测值、与论文的差值及两位小数是否一致。若数值匹配，结合输入、权重、提取器和计算检查，是**Avatar Forcing这一分支**已对齐的有力证据；不能自动证明DIM另一分支及P-FD／MSE也复现。

若不匹配，必须保留结果并定位可能原因：公开checkpoint绑定、avatar音轨、预处理、实际测试帧、提取后端或指标汇总。不能将差异全部归因于评分器，也不通过调设置凑原表。

入口：[生成](../../scripts/p2_avatarforcing_table4_generate.py)、[SPECTRE](../../scripts/p2_spectre_coefficients.py)、[评分](../../scripts/p2_avatarforcing_table4_score.py)、[接续控制器](../../scripts/p2_avatarforcing_table4_run.py)。此前已完成的[共同输入适配实测](../p2_vico43_unified_20261107/README.md)独立保留。
