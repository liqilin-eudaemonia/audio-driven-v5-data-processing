# Avatar Forcing Table4端到端检查

目的：用已取得的官方Avatar Forcing权重，按本篇论文的ViCo指标重新推理和评分，与Table4的Avatar Forcing行8个数值比较。验证的是该篇论文的评测分支；不把一次匹配扩大为DIM全部基线已复现。

论文证据：本地PDF第6页Table4，第5–6页预处理／推理，第14–15页D.5。SPECTRE、exp50／pose6、SID K15／9、逐系数时序方差平均与DIM的EMOCA／K40／20存在实质区别，不能只因Table4转引DIM旧行就视为所有设置相同。

方案选择：现有43条source-only适配视频只能作诊断；它们把user视觉置零，不能作为原版结果复现。此次保留旧结果，恢复官方encode_user_motion，使用speaker视频和音频；ViCo listening条件中的avatar音轨采用全零，作为明确推断记录。官方模型输入包括avatar音轨，但Table4未明确披露listening音轨处理，不能把该推断写成作者已确认。

官方test43完整保留；原始ViCo已裁脸媒体作为可用来源，按25fps处理。speaker走官方UserVideoProcessor裁剪路径，reference走官方单图预处理；整个说话人波形保留，长于30秒不截断。保留seed137及默认CFG2／1、10 NFE、KV cache；10 NFE不是10次候选，不复制DIM的GT择优规则。

SPECTRE官方权重从其quick_install.sh链接取得。仅执行系数编码部分，严格加载完整E_flame和E_expression，保留官方残差加回、五帧时域卷积和首尾两帧复制。验证完整与分块前向一致，再实际提取所有speaker／GT／generated帧。FLAME网格渲染不参与系数评分，不需要构造未使用的渲染器。

FD采用已验证的标准Fréchet函数；本篇式17均值项写法存在范数歧义，另报告未平方L2／L1的敏感性值，不择最接近论文者作为公式。SID使用K15／9，记录GT拟合与全局／逐片段聚合差异；Var采用论文明确的时间轴方差后沿特征轴平均；rPCC保留发布框架的拉平相关，并附逐系数差异检查。不按新分数调整裁剪、权重、子集或尺度。

资源：仅此前分配的node11 GPU7，新的小模型检查不恢复已暂停的DIM best10任务，不停止其他作业；先核对UUID和空闲内存，生成上限15%，至少48GiB空闲；提取上限8%，至少32GiB空闲。模型和数据分别冻结SHA，原产物只读。

交付：逐条视频／动作、GT覆盖、8项原文与实测对照、推断差异及一致性结论。若结果不一致，应报告不一致及可能来源，不能归因于评分器一个因素；若匹配，结合代码／输入一致性作为本篇设置对齐的证据，而非逻辑上证明所有论文协议相同。
