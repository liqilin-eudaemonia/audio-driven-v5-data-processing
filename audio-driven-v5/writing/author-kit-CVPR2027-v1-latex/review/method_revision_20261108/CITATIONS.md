# 引用替换清单

本版使用标准 `\cite{key}`；`references.bib`中的记录只是可编译的工作名称占位，不是可投稿的正式书目。**保留key，将整条记录替换为核实后的BibTeX**。不要把短工作名当作核实过的完整题名，也不由PDF文件名推定年份／会议。GLARE和OmniPersona保留提供稿状态。

| key | 定位名称 | 原文／入口 | 引用用途与注意事项 |
|---|---|---|---|
| `anytalker` | AnyTalker | [来源](<../../related/11577_AnyTalker_Data_Efficient(1).pdf>) | Emo-Acc、Emo-Div、DeltaResp来源 |
| `arig` | ARIG | [来源](<../40_ARIG_ICCV2025.pdf>) | 相关工作 |
| `avatarforcing` | Avatar Forcing | [来源](<../43_Avatar_Forcing_CVPR2026.pdf>) | 视觉比较；不由文件名推断正式出版年份 |
| `customlistener` | CustomListener | [来源](<../15_CustomListener_CVPR2024.pdf>) | 相关工作 |
| `difflistener` | DiffListener | [来源](<../34_DiffListener_CVPR2025.pdf>) | 相关工作 |
| `ditailistener` | DiTaiListener | [来源](<../41_DiTaiListener_ICCV2025.pdf>) | 相关工作 |
| `dystream` | DyStream | [来源](<../DyStream.pdf>) | 引言：双音轨条件；相关工作、视觉基线 |
| `elp` | Emotional Listener Portrait (ELP) | [来源](<../08_Emotional_Listener_Portrait_ICCV2023.pdf>) | 相关工作 |
| `gemma_pending` | Gemma (exact model reference pending) | 需确认实际Gemma版本后补条目 | 勿自行推定Gemma版本 |
| `glare` | GLARE | [来源](<../00_GLARE_local.pdf>) | 匿名提供稿；引言、相关工作、评价和视觉对照 |
| `infp` | INFP | [来源](<../35_INFP_CVPR2025.pdf>) | 引言：双音轨条件；相关工作 |
| `learningtolisten` | Learning to Listen | [来源](<../02_Learning_to_Listen_CVPR2022.pdf>) | 引言、相关工作 |
| `ltx_pending` | LTX (exact model reference pending) | 需确认实际LTX版本后补条目 | 勿自行推定LTX版本 |
| `mfrnet` | MFR-Net | [来源](<../06_MFR_Net_ACMMM2023.pdf>) | 相关工作 |
| `omnihuman15` | OmniHuman-1.5 | [来源](<../OmniHuman-1.5.pdf>) | 引言：给定语音的说话头像 |
| `omnipersona` | OmniPersona | [来源](<../OmniPerson.pdf>) | 匿名提供稿，文件名OmniPerson；引言、相关工作 |
| `omniresponse` | OmniResponse / ResponseNet | [来源](<../50_OmniResponse_NeurIPS2025.pdf>) | 引言、相关工作、数据和评价 |
| `qwen25omni` | Qwen2.5-Omni | 待补官方模型／论文条目 | 生成条件；使用3B版本 |
| `qwen3` | Qwen3 | 待补官方模型／论文条目 | 文本统计使用Qwen3-30B-A3B-Instruct-2507 |
| `qwen3omni` | Qwen3-Omni | 待补官方模型／论文条目 | 视听数据标注，非生成条件基座 |
| `react2023` | REACT2023 | [来源](<https://www.repository.cam.ac.uk/items/208fff1a-e416-4cd2-aa16-55eb01633574>) | 相关工作：多适当反应评价 |
| `reactface` | ReactFace | [来源](<../26_ReactFace_TVCG2025.pdf>) | 引言：多种合理反应；相关工作 |
| `realtalk` | RealTalk: Affective Faces for Goal-Driven Dyadic Communication | [来源](<https://realtalk.cs.columbia.edu/>) | 相关工作；不要下载另一同名talking-head论文 |
| `seamless` | Seamless Interaction | [来源](<https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset>) | 原始数据来源，不归为本文采集 |
| `speakervid` | SpeakerVid-5M | [来源](<../38_SpeakerVid_5M_2025.pdf>) | 相关工作、数据资源表 |
| `vico` | ViCo / Responsive Listening Head Generation (RLHG) | [来源](<../01_RLHG_ViCo_ECCV2022.pdf>) | 引言、相关工作、T1 |
| `whisper` | Whisper | 待补官方模型／论文条目 | speaker encoder使用large-v3 |

## 还没有补成引用的项目

- 数据需求段：按具体资源逐项补引用，不以泛泛的局限代替证据。
- LPIPS、FID、FVD、UTMOSv2、SyncNet、实际表情／活动检测工具：先核对实现与原论文，再增加正式条目。PSNR、SSIM按最终写法决定是否引用。
- Qwen→OVI尚为中文实验候选；OVI准确版本和论文待适配时核对。
- MLLM级联相关工作按作者确认保留中文待补，不为填满小节补造引用。

## 使用方法

1. 从作者官网、正式出版页面或论文arXiv页面导出BibTeX。
2. 用该条目替换`references.bib`中对应占位记录，将其key改回本清单中的稳定key。
3. 使用`latexmk -xelatex main.tex`重新编译，检查未定义引用。
4. 全部替换后删除PDF中“引用占位”作者备注。


## 2026-11-08 引言引用定位

下面对应作者11-08修订稿的进一步整理。**无需改正文中的key：取得正式BibTeX后，替换`references.bib`中同key的整个占位条目即可。** 工作名称占位不是正式书目信息，尤其不要据文件名推定出版状态。

| 引言位置／需支持的内容 | 已插入的引用键 | 原文定位与使用边界 |
|---|---|---|
| P1：倾听中的视觉反应与短声反馈 | `learningtolisten,omnipersona` | [Learning to Listen](../02_Learning_to_Listen_CVPR2022.pdf)第1页及§3；[OmniPersona](../OmniPerson.pdf)第2页行为段，明确举出uh-huh／mm及面部线索 |
| P2：根据目标人物给定语音生成画面 | `omnihuman15` | [OmniHuman-1.5](../OmniHuman-1.5.pdf)，用来说明目标语音驱动，不据此推断所有数字人仅有该能力 |
| P2：视觉倾听生成 | `vico,learningtolisten,glare`（逐个方法分别标注） | [ViCo](../01_RLHG_ViCo_ECCV2022.pdf)§3；Learning to Listen §3.1；[GLARE](../00_GLARE_local.pdf)§4 |
| P2：GLARE增加反应监督与评价 | `glare` | GLARE引言第1—2页及§4—5；保留其时间事件协议与本文整段协议的区别 |
| P2：从双音轨生成说话／倾听视频，目标声音是输入 | `infp,dystream`（分别标注） | [INFP](../35_INFP_CVPR2025.pdf)第1页Fig.1及§3；[DyStream](../DyStream.pdf)第1页与§3；此处描述视频生成模块，不能据此否认上游语音生成能力 |
| P2：已有视听回应及listener反馈 | `omniresponse,omnipersona`（分别标注） | [OmniResponse](../50_OmniResponse_NeurIPS2025.pdf)§3—5；OmniPersona第2页及§3—4；不说它们只处理接话或没有倾听评价 |
| P6：原始对话素材来源 | `seamless` | [Seamless原文入口](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset)；本文为再整理，不是原会话采集 |
| P6：视听描述使用的模型 | `qwen3omni` | 补Qwen3-Omni正式技术报告／模型对应引用；该cite不为本文完整性效果背书 |
| P7：同输入可有多种合理反应 | `learningtolisten,reactface` | Learning to Listen §3.1；[ReactFace](../26_ReactFace_TVCG2025.pdf)第1页摘要和任务定义；是非确定性／多适当反应基础，不是本文指标优越性结论 |

P3的应用理由来自本文输入条件；P4的挑战、P5的方法设计和P6的具体处理是本文研究内容，不用外部引用替代自己的实现与实验证据。P7仍是评价设计，不声称所有评分已执行。
