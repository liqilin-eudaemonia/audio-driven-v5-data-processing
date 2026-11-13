# ViCo文献协议：原论文、公开代码与本次执行对照

目标引用链：**DIM Table 1 → INFP Table 3 → Avatar Forcing Table 4**。不按新分数选择协议；已有本地DECA/BFM结果继续独立保留。

已核对正式发表PDF：DIM的Table 1在正式版第10页、本地预印本第11页，引用数值一致；INFP Table 3在第8页；Avatar Forcing Table 4在第6页。[逐行与逐格来源核验](LITERATURE_CELL_PROVENANCE.json)保存每个数值的原始出处和文件SHA。

| 核对项 | 原论文设置 | 公开代码的具体行为 | 我们的执行设置 | 依据及剩余差异 |
|---|---|---|---|---|
| 测试范围与OOD | DIM使用ViCo并沿用既有划分；INFP明确与DIM使用相同test | 按audio键读取；三路特征取最短长度，再限定5–1024帧 | 固定官方metadata的43个listener UUID，不混OOD；冻结源视频与完整时间轴 | DIM实验设置、INFP §5.2.2；作者实际保留的ID及有效帧段未公布。完整43条不等于已证实其发表时有效名单 |
| 提取器与权重 | DIM将所有基线统一到EMOCA表达空间 | 预处理明确读取EMOCA_v2_lr_mse_20目录的pose.npy和exp.npy | 从官方获取同名模型，固定checkpoint/config SHA，编码器644个状态张量严格加载 | DIM vico_preprocessing.py；EMOCA官方发布。DIM未提供其提取器配置、裁脸设置和权重SHA |
| 系数、分组与尺度 | expression50、pose6 | 顺序为pose6后接expression50，原始系数 | 保持同样顺序和原始尺度，不使用DECA缓存，不以GT统计反推缩放 | DIM补充表示说明、预处理和mymetrics.py。Pose6保留全局旋转与下颌参数，不冒充单独head3 |
| FD、P-FD、MSE | Fréchet距离、拼接speaker/listener后的Fréchet、逐帧MSE | 每片段计算后等权平均 | 直接调用固定版本作者函数，并与结构化输出对照；MSE保留作者原浮点汇总精度 | DIM补充A、mymetrics.py。我们由RGB重新提取，原表DIM基线主要直接评价原生预测系数 |
| SID | 补充文字描述各序列的聚类ID直方图熵 | 全测试GT拟合KMeans，exp40类/pose20类；全部预测帧的全局直方图，log2熵 | 按发布代码执行；记录依赖版本、随机状态与GT统计 | metrics/eval_utils.py。论文文字与代码聚合层次不完全一致，无法确认发表主表采用哪种 |
| Var | 文字描述listener动作的时序方差 | 全部帧及各维系数共同拉平后计算总体方差 | 沿用该公式，同时报告我们提取的GT统计与论文GT；不沿用旧表的平均时间方差 | mymetrics.py；两种Var定义不能混算或仅凭名字比较 |
| rPCC与speaker | 对真实speaker/listener运动关系计算残差相关 | 指标拉平后计算；默认数据加载器却将speaker动作替换为全1 | 提取真实speaker，仅供评分；常量导致未定义时记N/A | data_loader.py、mymetrics.py。全1默认分支无法解释原表有限rPCC，发表执行分支未确认 |
| 时间对齐 | 逐帧比较与双方协调 | tester去掉listener首帧，取speaker前T−1帧 | 明确沿该偏移：listener t1..T−1，对应speaker t0..T−2；不通过GT估计时延 | x_engine_pt.py evaluate_test_epoch。另保存原始帧索引与有效帧掩码；发表时具体时间约定未独立确认 |
| 生成与采样 | 主表未完整披露采样及择优规则 | tester生成10次，以完整56维GT Fréchet逐片段选择最优 | 固定复用已有seed137单次输出；不重新生成，不GT择优 | 发布tester；不能声称已匹配原表采样预算，也不能把公开默认路径直接当作已证实的发表设置 |
| 读出、训练与条件 | DIM基线在目标数据上重训；INFP/Avatar Forcing有各自原生条件与表示处理 | 原生系数送入动作评分；不同方法保留原生输入 | 固定联合8000/base-Qwen与SFT2752两版视频，经同一EMOCA提取；speaker视频只进入评分，不作为我们模型新增输入 | 原生系数与RGB反提取、训练集与输入条件不同，随表披露，不宣称等训练预算或完整原表复现 |

缺脸处理：保留原始帧号，缺失记为NaN，不做时序插值；两版模型共用同一有效帧掩码，报告每条覆盖。该处理是本次明确执行的选择，不能冒称作者未公布的过滤规则。

同时输出公开加载器的audio-key覆盖与5–1024帧范围诊断。由于没有作者HuBERT长度及实际EMOCA文件，诊断投影不是作者真实test；不据诊断分数改变预先固定的主范围。

**允许的表述：** 使用DIM公开预处理指定的EMOCA模型及其已发布指标实现，在固定ViCo test43上实测，并披露差异。文献行逐项注明原始结果来源。

**目前不能支持的表述：** 已严格复现DIM/INFP完整发表协议，或忽略上述差异直接计算跨来源提升率。评价组成对齐、发表执行细节完全一致，是两种不同程度的证据。

[固定协议](protocol.json) · [机器可读对照](ALIGNMENT.json) · [测试清单](TEST_MANIFEST.json) · [权重与来源](DOWNLOADS.json) · [正式发表版本核对](PUBLISHED_VERSION_COMPARISON.json)

补充版本证据：[历史核心方法核验](EMOCA_HISTORICAL_COMPATIBILITY.json)确认9个关键编码、系数拆分与裁脸方法，与2024-05-15（DIM初次代码发布之前）的EMOCA版本AST一致。这不等于确认作者的私有配置或提取参数。运行依赖另见[版本记录](ENVIRONMENT.json)。
