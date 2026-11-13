# ViCo43任务完成审计

按用户选择的方案实施：共同输入下本地实测我们／DyStream／Avatar Forcing，L2L人物迁移和文献数字分表。此处的完成指自动比较、可复核结果和展示交付完成；不声称完成新的L2L重训、原DIM表复现、人工研究或自主声音比较。

| 要求 | 当前证据与实际核验 | 结论 |
|---|---|---|
| 依据论文审计选择可落地比较 | DESIGN.md记录选择和边界；使用官方AF仓库507e3046及作者issue5支持的无user视觉条件；原论文引用不拼入本地分数 | 完成 |
| 正式ViCo测试范围 | INPUT_AUDIT.json及6项测试核对全部43个UUID，与官方test.csv一致；40个speaker时段／10个video_id，不合并不同listener，不含OOD或训练行 | 完成 |
| 共同输入与无目标泄漏 | 冻结生成代码只读取已核验speaker波形、相同listener首帧及合成零条件；全部43条输入与输出收据验证哈希；GT仅在外部评分阶段读取 | 完成 |
| 有权重的真实新基线 | 官方两核心权重严格覆盖所有非wav2vec参数；旧wav2vec weight_norm命名显式转换并逐张量相等。BUFFER_AUDIT.json另核对配置生成的mask，未加载MAE内的旧mask，保持官方named_parameters加载行为 | 完成 |
| 完整片段，不按GT选候选 | 43条正式视频完成；seed137每条一次，最短1秒及最长约35秒试跑与正式结果逐字节相同。直接完整波形调用，无CLI30秒截断 | 完成 |
| 原生与评分时间轴 | MEDIA_AUDIT.json完整解码86份视频，512/25原生及256/30评分副本尺寸／帧率／帧数正确，评分共16288帧；逐条最近时间戳映射核对 | 完成 |
| 共同图像评测 | 五项官方指标全43条；Inception权重SHA与原node8一致，ArcFace／代码／版本留档。独立复算CPBD帧加权，确认逐指标收据与最终表一致；新增GPU batch256、旧512明确披露 | 完成 |
| 动作空间和覆盖 | BFM/DECA不混排；新增43条两种表示均完整，漏检0。共同DECA含三种视频模型＋四套L2L原生模型共301条，逐个验证指纹、形状、有限值及独立L1，最大误差0；RTLCC42、RWTLCC41 | 完成 |
| 原有结果与暂停状态 | 原结果文件指纹不变，172条L2L输出完整保留，27749/3000/569步诊断未计入正式基线；本轮未启动任何L2L训练 | 完成 |
| 来源相关性与不确定性 | 每条可聚合动作指标给出配对差值，10000次整video_id聚类bootstrap，核对点估计及有效样本数；单seed和10来源限制披露，FID不冒充逐条指标 | 完成 |
| 完整可视化及解释 | 全43条按原顺序展示；总体／逐条距离、动作幅度／GT、L2L补充与CI齐备。最新HTML及data.json与REVIEW.json指纹一致 | 完成 |
| 实际浏览器行为 | BROWSER_AUDIT.json：430次媒体HEAD成功，代表片段实际播放和同步、原生/评分切换、seek、生成声音/静音切换、390px布局和CSV下载通过，JS错误0。此为功能抽查，不是43条逐条人工审看 | 完成 |
| 视觉适当性人工材料 | 86项全部43条×两baseline，左右均衡、顺序随机、标签隐藏；独立speaker上下文，两候选音轨静音。浏览器播放与空评分CSV导出通过，实际人工评分0份 | 工具准备完成；不宣称人评完成 |
| 可复跑与故障恢复 | 每指标flock、真实并发回归测试通过；一次重复CPU评分被受控停止，源码／协议／原日志归档，43条特征核验后复用，仅从评分恢复。无按分数更换模型或seed | 完成 |
| 文档与运行清理 | 本目录结果、JSON、CSV、截图和代码入口齐备；项目／evaluation总索引、原交接与旧结果入口新增指路。最终进程查询确认指定生成和两次控制器PID均不存在，pgrep无本轮ViCo任务 | 完成 |

## 结果解释边界

我们的PSNR/SSIM/FID/CSIM和表情参考距离较好；两baseline的CPBD更高。头旋转L1／Fréchet配对区间跨0，没有稳定的头姿领先证据。我们的表情运动方差约为GT的37%，应进一步由人工判断是否反馈不足，不能把参考距离下降自动升级为语境适当性。

两个基线均是公开模型在共同输入约束下的source-only适配，不生成listener声音；训练数据、预算和renderer未控制一致。L2L额外使用speaker motion，纯动作与视频反提取路径也不同。结果可作为论文中的ViCo视觉／动作补充实验，不等同原论文协议表或完整自主AV主实验。

对应机器可读证据：NUMERICAL_AND_MEDIA_AUDIT.json、MEDIA_AUDIT.json、BROWSER_AUDIT.json、IMAGE_BACKEND_AUDIT.json、FID_WEIGHT_AUDIT.json、SMOKE_FORMAL_REPRODUCIBILITY.json、SCORING_RESUME.json、FINAL_RUNTIME_OBSERVATION.json及TESTS.log。COMPLETE.json保存最终文件指纹。
