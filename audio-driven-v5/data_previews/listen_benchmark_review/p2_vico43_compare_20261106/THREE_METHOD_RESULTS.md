# ViCo test43：三种方法统一对照

[打开统一页面](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/) · [完整指标CSV](all_methods_metrics.csv) · [页面数据JSON](data.json)

三种方法为联合模型8000步、DyStream、Learning2Listen。前两种提供真实推理视频，可与speaker及listener GT同步播放；Learning2Listen仅提供四套人物模型的动作参数，没有公开写实renderer，图像指标标N/A。

| 方法 | PSNR ↑ | SSIM ↑ | 图像FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| 联合模型 · 8000步 | 19.668293 | 0.662447 | 30.116312 | 0.596836 | 0.139126 |
| DyStream · listener适配 | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |
| Learning2Listen（四套） | N/A | N/A | N/A | N/A | N/A |

新增共同DECA动作评分：联合模型、DyStream的视频统一反提取DECA，与四套L2L原生动作在同一50＋3＋3空间评分。总体及逐片段结果均已接入页面，完整原始数值见 `experiments/common_method_comparison_20261106/vico_deca/RESULTS.json`。原生动作与视频反提取的输出路径、训练条件仍不同，不能宣称是DIM原表复现。

BFM运动结果在页面并列表格中逐项对照；FD与FID_delta_fm展示乘100。Learning2Listen的DECA表情50维、头部旋转3维及下颌旋转3维单独列出，未与BFM混算或跨空间排名。RTLCC覆盖42条，RWTLCC覆盖41条，其余指标43条；不适用的短片段标N/A。

同步播放器默认原生512×512／25FPS，可切换实际256×256／30FPS评分副本。试听可以选择speaker、联合模型生成声音、二者各50%混合或静音。DyStream不生成listener声音，视频自带speaker上下文音轨始终静音。

Learning2Listen四套人物权重均保留，无按GT择优选择；可在同一片段切换参数曲线与逐条指标。它额外使用speaker视频动作，相关输入与4t音频对齐修正沿用已确认协议。原模型、推理产物和评测JSON不变，本次只重新组织展示。

完整原始评分与资源限制见原评测目录 `experiments/p2_compare_vico43_20261106/RESULTS.md`。
