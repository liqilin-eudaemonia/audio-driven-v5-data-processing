# 原文标注指导推理：实现与GPU验证

按确认方案保留原文及时间，合并音频描述/发声文本。24条原train样本和24条test样本都已导出指导文本并完成Gemma缓存，不修改原数据或固定prompt缓存。

原train24的文本长度分档：18条1024、5条2048、1条4096；test24：16条1024、4条2048、4条4096。分档是padding长度；实际最长分别2931和2872 tokens。均检查完整token ID序列，没有截断。

**工程验证完成：四组权重各2条短样本（8条）、E0F一条4096档长文本、一条E0固定prompt重放，共10条输出。不是四组全部24条的正式效果评测。**

| 输出 | 配置 | 新音轨RMS | 对应旧固定prompt RMS | 视频与原文 |
|---|---|---:|---:|---|
| E0-0 | guided | 0.06495578 | 0.00014678 | [视频](validation_outputs/E0-0/generated.mp4) · [音频](validation_outputs/E0-0/listener.wav) · [prompt](validation_outputs/E0-0/prompt.txt) |
| E0-1 | guided | 0.07427492 | 0.00003500 | [视频](validation_outputs/E0-1/generated.mp4) · [音频](validation_outputs/E0-1/listener.wav) · [prompt](validation_outputs/E0-1/prompt.txt) |
| E1-0 | guided | 0.07632405 | 0.00067931 | [视频](validation_outputs/E1-0/generated.mp4) · [音频](validation_outputs/E1-0/listener.wav) · [prompt](validation_outputs/E1-0/prompt.txt) |
| E1-1 | guided | 0.04581055 | 0.00003901 | [视频](validation_outputs/E1-1/generated.mp4) · [音频](validation_outputs/E1-1/listener.wav) · [prompt](validation_outputs/E1-1/prompt.txt) |
| E0F-0 | guided | 0.06387958 | 0.00016752 | [视频](validation_outputs/E0F-0/generated.mp4) · [音频](validation_outputs/E0F-0/listener.wav) · [prompt](validation_outputs/E0F-0/prompt.txt) |
| E0F-1 | guided | 0.10946313 | 0.00005215 | [视频](validation_outputs/E0F-1/generated.mp4) · [音频](validation_outputs/E0F-1/listener.wav) · [prompt](validation_outputs/E0F-1/prompt.txt) |
| E1F-0 | guided | 0.05704784 | 0.00009589 | [视频](validation_outputs/E1F-0/generated.mp4) · [音频](validation_outputs/E1F-0/listener.wav) · [prompt](validation_outputs/E1F-0/prompt.txt) |
| E1F-1 | guided | 0.07636647 | 0.00003470 | [视频](validation_outputs/E1F-1/generated.mp4) · [音频](validation_outputs/E1F-1/listener.wav) · [prompt](validation_outputs/E1F-1/prompt.txt) |
| long-4096 | guided | 0.01909624 | 0.00016522 | [视频](validation_outputs/long-4096/generated.mp4) · [音频](validation_outputs/long-4096/listener.wav) · [prompt](validation_outputs/long-4096/prompt.txt) |
| fixed-replay | fixed | 0.00014678 | 0.00014678 | [视频](validation_outputs/fixed-replay/generated.mp4) · [音频](validation_outputs/fixed-replay/listener.wav) · [prompt](validation_outputs/fixed-replay/prompt.txt) |

固定prompt重放不是逐位一致：音频RMS差值（波形差的RMS）为9.9128784e-07，最大逐采样差9.059906e-06；视频像素MAE 0/255，最大差0/255。原配置、权重、正负文本与seed一致；保留实际差异，不声称位级复现。

两条短样本的指导版音频电平明显增加。RMS仅说明幅度，不证明发声内容、音色、时间遵循或与speaker的适当性；后续需试听并独立核验。指导使用目标标注，全部明确标记annotation-guided diagnostic，不能混入旧source-only E0/E1成绩。

- [完整验证记录](validation_summary.json)、[全部prompt及缓存校验](cache_validation.json)、[验证任务](validation_jobs.json)。
- [逐样本prompt](prompts/)，`.txt`是实际Gemma输入，`.json`包含完整原标注和每行来源。
- [实现与命令](../../related/P2_原文标注Gemma指导推理实现.md)。
- 35项新增CPU测试及37项现有评测入口测试通过。没有修改训练实现、权重或已有HTML。
