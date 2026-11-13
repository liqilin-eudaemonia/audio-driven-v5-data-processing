# 现在即可引用的已完成实测

此处汇集11月7日已封存的43条共同输入适配结果；本轮只读取原成绩，没有重跑或改变公式。公开权重没有用本轮500h重训。

| 方法 | PSNR ↑ | SSIM ↑ | 图像FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| Ours / joint8000 | 19.668293 | 0.662447 | 30.116312 | 0.596836 | 0.139126 |
| DyStream / public pretrained / listener adaptation | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |
| Avatar Forcing / public pretrained / zero speaker visual | 16.183142 | 0.572531 | 39.152789 | 0.466331 | 0.190193 |

固定ViCo test43、listener首帧与speaker音频、每条seed137单次生成。DyStream和Avatar Forcing的listener自身音轨为零；本表Avatar Forcing的speaker视觉条件为零。图像FID不是DIM的表情／姿态FD。

[CSV](existing_image_metrics.csv) · [源成绩与协议](../p2_vico43_unified_20261107/RESULTS.md) · [本轮统一EMOCA进度](README.md)

Learning2Listen四套人物迁移模型的独立DECA动作分数：

| 人物权重 | 已生成条数 | 表情L1 ↓ | 头旋转L1 ↓ | 下颌L1 ↓ |
|---|---:|---:|---:|---:|
| conan | 43 | 0.288010 | 0.148715 | 0.040810 |
| fallon | 43 | 0.304757 | 0.179173 | 0.048678 |
| stephen | 43 | 0.267368 | 0.161168 | 0.047367 |
| trevor | 43 | 0.261020 | 0.181103 | 0.033099 |

L2L为公开人物权重的迁移测试，额外使用speaker motion，没有在ViCo重训；四套全部报告，没有按test挑选一套。只有动作输出，不能填写实RGB指标或与EMOCA系数混排。
