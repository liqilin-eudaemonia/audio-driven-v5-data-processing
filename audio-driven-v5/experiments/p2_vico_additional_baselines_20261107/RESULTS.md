# ViCo43：新增基线比较结果

[全部43条同步检查](http://localhost:8876/listen_benchmark_review/vico_more_20261107/) · [完整JSON](COMPARISON.json) · [论文PDF](paper_tables/preview.pdf) · [逐条完整动作CSV](paper_tables/all_motion_per_clip.csv)

固定43个正式test pair、10个来源；全部保留，一次生成。原生动作、视频反提取及DualTalk迁移代理分表，不能混合数值或推断统一排名。所有旧结果只读复用。

| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ | 表情L1 ↓ | 头旋转L1 ↓ | 表情方差／GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours / base Qwen | 19.668293 | 0.662447 | 30.116312 | 0.596836 | 0.139126 | 0.153944 | 0.096661 | 37.2% |
| DyStream adapted | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 | 0.212086 | 0.106644 | 46.7% |
| Avatar Forcing adapted | 16.183142 | 0.572531 | 39.152789 | 0.466331 | 0.190193 | 0.224490 | 0.104874 | 59.1% |
| Ours / SFT2752 | 19.335550 | 0.655867 | 29.426238 | 0.580254 | 0.137383 | 0.167187 | 0.087148 | 71.1% |
| ListenFormer local retraining | 18.887633 | 0.641760 | 30.842915 | 0.602755 | 0.193178 | 0.160342 | 0.101508 | 3.2% |

图像指标沿用官方帧加权和集合FID；动作距离／方差逐pair等权。幅度过小或过大都可能不合适，低参考距离不等于反馈更自然；尚无人工适当性评分。

## 已完成的新训练及权重选择

- ListenFormer local retraining：完成200轮；验证集选择第1轮／869步权重。test43未参与统计和选权重。

ListenFormer local retraining 的共同视频读出表情方差仅为GT的3.2%，存在明显动作不足。较低的参考距离不能单独作为优于其他方法的证据；当前保留验证集选中的权重，不根据test结果换权重。

## 与SFT2752的来源聚类配对差

差值为SFT2752−新增方法；距离指标为负表示SFT2752较低。95%区间按来源video_id聚类，10000次、seed137；区间跨0时不作稳定优劣结论。方差只辅助解释。

| 对照 | 指标 | 平均差 | 95%区间 | pair／来源 |
|---|---|---:|---|---|
| ListenFormer local retraining | L1/exp | 0.0068443 | [0.0006281, 0.0225964] | 43／10 |
| ListenFormer local retraining | L1/head_rotation | -0.0143595 | [-0.0241606, 0.0037857] | 43／10 |
| ListenFormer local retraining | Frechet/exp | -0.8282060 | [-1.1385974, -0.0840875] | 43／10 |
| ListenFormer local retraining | Delta_Frechet/exp | -0.0505394 | [-0.0843339, -0.0336848] | 43／10 |
| ListenFormer local retraining | V_D/exp | 0.0150859 | [0.0106855, 0.0178620] | 43／10 |

## 其他读出及适用边界

L2L四个人物权重全部逐条提供原生曲线与指标，不按test挑选人物；DualTalk提供官方权重的43条动作迁移代理，neck／global head几何尚未验证，因此不进入共同头部或RGB优劣主张。
RLHG／ListenFormer额外接收speaker动作和listener首帧系数，并在ViCo本地重训；我们的生成器与两种既有视频baseline训练条件不同。本表不是等训练预算或作者论文原表复现。
CSV空白表示不适用（N/A），覆盖数保留在JSON；RTLCC/RWTLCC分别仅支持42／41条，不填充短片段。
[条件与训练口径CSV](paper_tables/conditions.csv) · [BFM视频反提取表](paper_tables/bfm_video.tex) · [独立BFM原生表](paper_tables/bfm_native.tex) · [L2L表](paper_tables/l2l_native.tex) · [DualTalk代理表](paper_tables/dualtalk_proxy.tex) · [方差表](paper_tables/rgb_motion_amplitude.tex)
