# 原文、补充材料与新修正的证据

本地PDF页序用于定位，不以会议印刷页码代替。主文对应文件与指纹见INVENTORY；作者模板排除，两份相同DiffListener只计一篇。

## 取得并审阅的额外文字材料

| 材料 | 本地文件与官方来源 | 审阅结果 |
|---|---|---|
| L2L完整arXiv版 | [17页PDF](web/L2L_arxiv.pdf) · [arXiv](https://arxiv.org/abs/2204.08451) | 附录p12–17；四listener聚合、分人物Tables3–6、TLCC7、LFI原split motion-only重训及GT best-of200专项分析 |
| CustomListener | [5页补充](web/CustomListener_supp.pdf) · [CVF](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Liu_CustomListener_Text-guided_Responsive_CVPR_2024_supplemental.pdf) | 全5页；RealTalk Tables5/6明确重训，GPT生成文本先验、ELP官网可视化比较 |
| DiffListener | [4页补充](web/DiffListener_supp_07114_supp.pdf) · [官方ZIP](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wang_Diffusion-based_Realistic_Listening_CVPR_2025_supplemental.zip) | 全4页；A.3明确RLHG/PCH去emotion标签重训，Table6额外26人用户研究，Table5为自身条件消融 |
| INFP | [3页补充](web/INFP_supp_9131_supplementary.pdf) · [官方ZIP](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhu_INFP_Audio-Driven_Interactive_CVPR_2025_supplemental.zip) | 全3页；Table1自身数据/记忆库消融，ViCo与HDTF定性方法同正文，无额外独立定量baseline |
| ARIG | [5页补充](web/ARIG_CVF_supp.pdf) · [CVF](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Guo_ARIG_Autoregressive_Interactive_ICCV_2025_supplemental.pdf) | 全5页；Table2为RealTalk20例25人研究；内部motion输入262维，不能与生成后反提70维评价混同 |
| DiTaiListener | [新版3页补充](web/DiTaiListener_CVF_supp.pdf) · [CVF](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Siniukov_DiTaiListener_Controllable_High_ICCV_2025_supplemental.pdf) | 全3页；比本地15页稿多Table6长视频策略、Table7 CTM消融；文本标注来源为listener视频+speaker audio，测试时文本获取方式仍未明确 |
| ReactFace作者版与当前本地版 | [作者版](web/ReactFace_author_pdf.pdf) · [版本核对](web/REACTFACE_VERSION_RECHECK.json) | 初始本地文件0页；工作区后来改为18页有效文件，本审计未改源PDF。Table2/8/9已重新核对，两文件非字节相同；p15 RLD额外表必须保留 |

ELP官方CVF ZIP的目录只有视频，没有PDF/HTML/text补充。目录与抽取文件指纹记录在[CVF ZIP清单](web/CVF_ZIP_DOCUMENTS.json)。使用HTTP range取得ZIP目录与科学文档，没有下载全部视频包。INFP最初在读取AppleDouble元数据伪PDF时遇错，真正3页补充已成功提取和阅读；记录已区分这两件事。

GLARE、RLHG、DIM、SpeakerVid、DiTai本地版、Avatar Forcing、DyStream及OmniPersona的相应附录已嵌在本地PDF中；TABLE_CENSUS按阶段登记，不因编号重复遗漏DIM附录Table1/2。

## 几处影响实验安排的原文事实

- **GLARE输入不能只简写为audio＋首帧。** p6还写previous listener motion；p16明确为缓解静态输出加入Tref，推理全部实验用同一个固定视频clip。它可能是固定先验，而非逐例目标真值，但来源／身份未报告，不能替作者断言完全同输入或用了GT。p16的baseline训练650epochs与p17 Table7 final850epochs也应分别记录。
- **DiffListener的RLHG/PCH不是原始含attitude设置。** 官方补充A.3明确在两数据集上均移除emotion标签后重训。引用其结果时同时说明实现与条件变化。
- **DIM Fig4的L2L不是普通行。** 图注明确是CANDOR预训练版；定量引用普通L2L和图中视觉例的来源不能互相替代。
- **ELP只在部分单元标直接引用。** ViCo的RLHG两个FD格标*；L2L数据集的L2L行在FD/V-D/SI-D/RPCC的expression/head格标*；其余WTLCC/STS/blink没有相同声明。表格来源与公式／尺度是否一致还应分开。
- **OmniResponse披露了两个具体版本。** p8注明Claude-Sonnet-4为2025-05-14、DeepSeek-R1为2025-05-28，后者应核[R1-0528官方模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)。其余未报告的snapshot、prompt或baseline训练细节不猜补。

## 图中方法身份与不可识别项

ECHO Fig5（p17）明确比较INFP、ARIG、StreamAvatar，但只提供定性样例。对应StreamAvatar为[Zhiyao Sun等人的原项目](https://streamavatar.github.io/)，输入双方已知audio＋reference，不是检索中同名的DyStream/AROD蒸馏仓库。ECHO没有可给StreamAvatar填入的ViCo数值。若另引原论文数字，应引[StreamAvatar原文](https://arxiv.org/html/2512.22065v1)Table1或EMTD110例的Table4，并保留其已知目标audio、全身视频与数据范围。

ECHO Fig3是栅格图片，文字提取没有读出方法行。[原图渲染证据](web/ECHO_p11_figure3.png)显示 **Per-FLOAT、DIM、DualTalk**，不能根据主定量表替换成Avatar Forcing。Per-FLOAT没有对应实现说明，图中[51]/[37]/[31]与当前文献表不一致，因此保留原标签及“身份未明确”，不擅自并入FLOAT或其他模型。

OmniHuman-1.5的CS-A/B/C/D是作者匿名商业比较；Table2(c)还有未命名的general Base Model。OmniPersona附录A提到未命名concurrent-work视频。没有可核实身份就记录未知，不猜具体商业系统，也不将背景引用的所有方法递归扩张成必须测试的无限集合。

OmniPersona Table1是能力描述，不是性能实验。其“Joint A&V”标记不能直接替代我们按任务输入输出的分类：OmniResponse确实同时输出音频／视觉，但架构与联合扩散不同。表中判断依赖作者定义，不能据一个✗断言它不生成音视频。

名称计数使用规范化条目：ViCo/RLHG、PCH/PCHG、Avatar Forcing/AvatarForcing分别合并，INFP的两个作者重实现与CANDOR预训练变体仍分别保留。DiTaiListener Table2把第一行写成RLFG且引文编号与L2L相同，正文baseline名单则是RLHG；本清单据正文归入RLHG并保留这一拼写／编号冲突，不宣称发现了一个已确认的独立RLFG模型。原图／表的拼写不应被当作另一套可下载模型。

## 逐行来源如何查

[ROW_PROVENANCE.csv](ROW_PROVENANCE.csv)逐条列出方法、表/图/正文位置、数据、页码、来源类型、源论文与条件说明；[JSON](ROW_PROVENANCE.json)保留ELP按指标单元拆分的结构。`numerical_match_only`和`unreported`不会被替换为确认引用、官方权重推理或从零训练。有公开代码／权重是资源事实，不是原论文实际用过该权重的证据。
