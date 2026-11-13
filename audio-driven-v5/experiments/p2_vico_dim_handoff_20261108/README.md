# ViCo / DIM 工程验收与对齐交接

2026-11-08：用户要求继续完成工程，之后给出需要人工double-check的清单。本轮核对发现，前述工程已在[文献协议实测目录](../p2_vico_literature_protocol_20261108/README.md)完成。此前旧审计表仍写“尚未部署EMOCA”，造成状态误报；现以本页和真实验收结果为准，历史记录不覆盖。

**工程已完成并复验通过；严格对齐DIM已发表原表仍未确认。** 两个状态分别记录。

- [需要你double-check的具体清单](DOUBLE_CHECK.md)：5组作者协议问题＋2项论文报告选择，包含提问方式、所需材料、证据与确认后的动作。
- [本轮机器验收](VERIFICATION.json)：48份原封存文件、6个实现文件、14个原始来源文件、172个输入视频、172组动作数组及checkpoint指纹检查通过。两版test43的548个指标／逐片段数值重算最大误差0；与原作者函数的16个汇总数一致。没有新占GPU或重训。
- [GT范围诊断](GT_SCOPE_DIAGNOSTICS.json)：GT统计差异在全部16,288帧上依然存在，不能归因于7帧生成漏脸的共同掩码。
- [已有完整结果](../p2_vico_literature_protocol_20261108/RESULTS.md)、[CSV](../p2_vico_literature_protocol_20261108/paper_tables/all_metrics.csv)、[论文表格预览](../p2_vico_literature_protocol_20261108/paper_tables/preview.pdf)。

## 已完成的工程

固定ViCo官方test43个独立listener UUID。speaker、GT、联合8000/base-Qwen、联合8000/SFT2752共四路，每路16,288帧，总计172组特征、65,152帧。使用DIM预处理点名的`EMOCA_v2_lr_mse_20`，严格加载644个编码器张量，系数为原始pose6＋exp50。

两版模型均已计算FD、P-FD、MSE、SID、Var、rPCC，exp／pose分组。评分沿DIM固定版本公开函数，listener去首帧与speaker前T−1帧对应。43条均保留，去首帧后16,245个候选帧，共同有效16,238帧；base生成有7帧未检出脸，另三路无漏脸。失败不插值、不当零误差，缺项单列。

两版表情／姿态FD分别为25.89253／0.08714和24.33909／0.06182。它们是本地实际测量值，不能据此宣称已在完全相同发表设置下超过引用基线。

## 剩余工作为什么需要外部确认

GT表情Var实测1.45611，DIM论文0.93；姿态Var实测0.02327，论文0.01。完整GT帧、去首帧、共同有效掩码以及可见加载器投影均已检查，仍没有证据证明我们复现了作者实际参考数据处理。要进一步定位，优先需要作者的实际测试清单、GT特征／提取配置以及主表评测配置。

具体尚未证实的项目见double-check清单；不会通过调缩放、选测试子集、按GT挑模型或猜测私有配置使数字接近。

## 只读复验

在项目根目录运行，输出仅写本交接目录，不修改原始封存结果：

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  compare/.envs/vico_more/bin/python \
  experiments/p2_vico_dim_handoff_20261108/verify_handoff.py
```

七类整段事件评测是独立工作；它的4对pilot和人工校准不用于判断本次ViCo/DIM工程是否完成。
