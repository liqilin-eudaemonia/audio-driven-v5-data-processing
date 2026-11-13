# ViCo test43：统一本地实测

[完整43条同步视频与评分](http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/) · [完整结果](RESULTS.json) · [画质CSV](image_metrics.csv) · [逐条动作CSV](deca_per_clip.csv) · [设计与边界](DESIGN.md)

三种视频方法使用全部43条正式test、10个来源对话、40个不同speaker时段；保留不同listener UUID。我们模型固定为先前指定的联合8000，两个baseline使用官方核心模型的source-only适配。每条seed137、一次采样，不按GT择优。原生512/25，评分256/30。

| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| 我们 · 联合8000 | 19.668293 | 0.662447 | 30.116312 | 0.596836 | 0.139126 |
| DyStream · listener适配 | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |
| Avatar Forcing · source-only适配 | 16.183142 | 0.572531 | 39.152789 | 0.466331 | 0.190193 |

以下为同一DECA表情50、头旋转3、下颌3的独立动作比较；不是DIM的EMOCA原表。DECA原始弧度，无平移；不混入BFM数值。

| 方法 | 表情 L1 ↓ | 头旋转 L1 ↓ | 下颌 L1 ↓ | 表情 Fréchet ↓ | 头旋转 Fréchet ↓ |
|---|---:|---:|---:|---:|---:|
| 我们 · 联合8000 | 0.153944 | 0.096661 | 0.016081 | 2.168830 | 0.084991 |
| DyStream · listener适配 | 0.212086 | 0.106644 | 0.019132 | 3.922448 | 0.065514 |
| Avatar Forcing · source-only适配 | 0.224490 | 0.104874 | 0.019862 | 4.548128 | 0.065957 |

## 动作幅度参考

方差只反映运动幅度，不按高低直接排名；应同时检查不足与过度。

| 来源 | 表情方差 | 头旋转方差 | 下颌方差 |
|---|---:|---:|---:|
| Listener GT | 0.02222699 | 0.00996665 | 0.00024011 |
| 我们 · 联合8000 | 0.00825794 | 0.00200818 | 0.00005330 |
| DyStream · listener适配 | 0.01037899 | 0.00099086 | 0.00006846 |
| Avatar Forcing · source-only适配 | 0.01312607 | 0.00206989 | 0.00018900 |

## BFM70独立补充

[三模型全部BFM指标CSV](bfm_metrics.csv)。以下FD是原始L1，未乘100；exp64、angle3、translation3，与DECA分开。

| 方法 | FD/L1 表情 ↓ | FD/L1 角度 ↓ | FD/L1 平移 ↓ | RTLCC差异/exp ↓ | RWTLCC差异/exp ↓ |
|---|---:|---:|---:|---:|---:|
| 我们 · 联合8000 | 0.131129 | 0.102194 | 0.064931 | 0.225741 | 0.292209 |
| DyStream · listener适配 | 0.154563 | 0.112404 | 0.070048 | 0.230798 | 0.298834 |
| Avatar Forcing · source-only适配 | 0.170757 | 0.108340 | 0.081859 | 0.247978 | 0.291402 |

## 配对差值与不确定性

差值为“我们−baseline”；下面距离指标为负表示我们较低。95%区间使用10000次、seed137，按原始video_id整组重采样，保留同来源片段的相关性。区间仅反映固定模型/固定seed下测试来源的不确定性，不包含重新训练或多seed采样差异；多项指标属于探索性报告。

| baseline | 指标 | 配对平均差 | 95%区间 | 有效pair/来源 |
|---|---|---:|---|---|
| DyStream · listener适配 | L1/exp | -0.058142 | [-0.068002, -0.045073] | 43/10 |
| DyStream · listener适配 | L1/head_rotation | -0.009983 | [-0.034086, 0.003735] | 43/10 |
| DyStream · listener适配 | Frechet/exp | -1.753618 | [-2.108467, -0.984106] | 43/10 |
| DyStream · listener适配 | RTLCC/exp | -0.011133 | [-0.031050, -0.000470] | 42/10 |
| DyStream · listener适配 | RWTLCC/exp | -0.013337 | [-0.023958, -0.001866] | 41/10 |
| Avatar Forcing · source-only适配 | L1/exp | -0.070546 | [-0.109630, -0.049738] | 43/10 |
| Avatar Forcing · source-only适配 | L1/head_rotation | -0.008212 | [-0.029158, 0.003299] | 43/10 |
| Avatar Forcing · source-only适配 | Frechet/exp | -2.379298 | [-3.706454, -1.790669] | 43/10 |
| Avatar Forcing · source-only适配 | RTLCC/exp | -0.025915 | [-0.034760, -0.002114] | 42/10 |
| Avatar Forcing · source-only适配 | RWTLCC/exp | -0.008420 | [-0.014022, 0.004217] | 41/10 |

## 适配、覆盖与解释

Avatar Forcing：真实speaker进入user音频，自身音轨全零；user视觉为条件空间零向量，无speaker视频输入。模型保留官方CFG2/1、nfe10和KV cache。已裁好的共同首帧不二次裁脸；完整波形直接调用模型，避免官方CLI30秒上限截掉最长约35秒片段。旧wav2vec weight_norm参数名已显式映射，加载后逐张量相等；无监督、无训练更新。
DyStream：真实speaker进入other，自身音轨全零，同一首帧、crop=False、官方5步。两者均只生成视频，音频指标N/A。网页试听speaker仅为上下文；只有我们模型有生成listener音轨。
图像指标沿用同一官方ViCo实现和聚合。动作逐pair等权，RTLCC42条、RWTLCC41条，其余43条；短片段不补帧。25→30最近时间戳映射重复帧会影响动态指标，逐条映射和尾端误差已保留。
这里RTLCC/RWTLCC是speaker–GT与speaker–prediction相关曲线的绝对差异，越低越好；不是原始相关系数，也不是DIM的rPCC。图像GPU批次为256，旧结果为512；同一eval-mode网络与权重，未声称不同batch的浮点结果逐字节一致。[执行环境与批次](IMAGE_BACKEND_AUDIT.json)。
L2L四套官方人物迁移动作仍全部保留在RESULTS.json和逐条CSV，额外使用speaker motion，未在ViCo重训，也没有写实renderer。暂停的27749/3000/569步诊断均未进入比较。文献表保留在原交接入口，未拼入这张本地表。
训练数据、预算和renderer不同，因此本实验是已有模型在共同输入上的迁移比较，不能声称受控重训或严格复现原论文表。ViCo只有单一GT，图像/动作距离不能独立证明回应是否适当，更不能证明自主声音生成优于不生成声音的baseline。

## 本轮能支持什么

在这43条固定输入上，我们的画质/身份匹配和表情参考距离较好，表情L1配对差值区间对两个baseline均为负。头旋转的L1和Fréchet差值区间都跨0，尚无稳定的头姿优势；两种baseline的CPBD更高。
我们的表情方差为GT的37.2%，显示运动幅度偏小。较低的参考距离需要与动作幅度及人工适当性共同解释，不据此宣称回应更充分。

[86项A/B视觉适当性盲评](http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/blind.html) · [人工协议与空评分表](human_review/PROTOCOL.md)。已准备全部43条×2组比较，模型名隐藏、位置随机且均衡；目前人工评分0份。正式评审仅使用盲评入口，不提前查看带模型名的对照页。

[Avatar Forcing官方仓库](https://github.com/TaekyungKi/AvatarForcing) · [作者关于视觉置零的说明](https://github.com/TaekyungKi/AvatarForcing/issues/5)
