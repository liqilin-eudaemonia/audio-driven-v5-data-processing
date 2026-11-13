# 引用哪篇、必须对齐什么协议

原论文数字的引用权与实验可比性分开。可引用公开论文并注明来源，但不能在自己的J短pair结果旁直接计算提升率。缺模型权重也不意味着只能复制数字：有完整训练代码可以重训，有部分代码可明确标注重实现；两者都需要额外实验验收。

| 要参考的方法／结果 | 优先引用源头 | 原表的共同协议 | 加入我们结果前需补齐 |
|---|---|---|---|
| ViCo原始RLHG、Random/Simulation | ViCo原论文Table3/4，PDF p10–11 | ViCo Dtest/Dood；BFM expression64＋angle3＋trans3；动作FD为L1×100；独立renderer质量 | 官方划分、提取器、尺度、态度及renderer；不能以Fréchet实现替代FD |
| PCH官方challenge成绩 | PCH Table1及官方leaderboard，p2 | ViCo challenge talking/listening不同任务；PSNR/FID/LMD或ExpFD；其他行是团队提交 | challenge实际测试范围、服务器评分与渲染；不把团队名当公开模型 |
| MFR-Net原始数字 | MFR-Net Table1/2，p6 | ViCo test/OOD分态度，L1 FD×100；图像汇总含两子集；renderer有身份训练 | 具体子集与聚合、身份特征、裁剪；CustomListener/DiffListener/ListenFormer中同值应溯回MFR |
| CustomListener ViCo | CustomListener Table1/2，p6 | BFM70维、L1 FD×100、RTLCC/RWTLCC、FIDΔfm/SND；文本控制 | 文本是GT描述还是speaker预测文本、4s窗口/时移、test/OOD和renderer |
| CustomListener RealTalk | 官方CVF补充Table5/6，补充p2–3 | RealTalk上重训PCH/RLHG/L2L；自动情绪转换态度；与ViCo同70维指标体系 | 对应RealTalk划分与人脸提取；不能引用其ViCo值来填RealTalk |
| DiffListener | DiffListener Table1 RealTalk / Table2 ViCo，p6–7 | RealTalk1489/407；ViCo398/43/42；统一评价3DMM，L1 FD/FIDΔfm/SND及图像指标 | motion内部MediaPipe与最终评价表示不同；512裁脸；其MFR/Custom行引用原论文，保留来源脚注 |
| ELP原模型 | ELP Table1，p5–6 | ViCo与L2L分开；ELP自己的expression100体系；眨眼独立；FD命名与L1定义不一致 | 必须核对实现公式和尺度；不能搬到DIM EMOCA表里；带星号只引用部分单元 |
| ELP在共同EMOCA下的重实现 | DIM Table1（ViCo）/Table2（LM_Listener），p11 | DIM作者ELP重实现，与RLHG/L2L统一EMOCA56维训练；Fréchet FD/P-FD/MSE/SID/Var/rPCC | 结果应称DIM作者重实现ELP，而非ELP原表；精确测试及采样仍缺口 |
| RLHG/L2L在ViCo的EMOCA数字 | **DIM Table1**，p11 | expression50＋pose6；在目标数据重训；L2L普通与CANDOR预训练†不同 | 精确有效test列表、speaker输入、候选数及是否GT择优；旧公开代码审计不能证明主表已对齐 |
| L2L官方原始模型与数据成绩 | L2L Table1，p6–7；17页arXiv补充p13–16 | DECA50 expression＋jaw3，另head rotation3；人物模型分别训练，70/20/10划分；补充明确跨四个listener聚合，六频道另两人因数据不足排除；FD/P-FD/SI/Var/PCC | 固定人物与聚合；四套人物权重跨ViCo零样本不是DIM所报ViCo重训结果；补充Fig7的GT best-of200是专项分析，不能移植为主表默认策略 |
| LFI在其原数据上的对照 | L2L arXiv补充D正文，p14；LFI方法引用IVA2020原文 | 原LFI facial annotation及原split，motion-only；L2L作者重训LFI与自身m消融；FD/P-FD/Var | 无原音视频，不包含目标声音生成；引用这些值须说明L2L作者重训，不能称原LFI官方checkpoint结果 |
| DIM自身 | DIM Table1/2；speaker任务Table4，p11、14 | listener为ViCo/LM_Listener；speaker为BiWi，EMOCA转换FLAME5023顶点，LVE/FDD | 不把BiWi说话任务当倾听任务；renderer对test人物微调要披露 |
| INFP原始listener结果 | **INFP Table3**，p8 | 沿DIM的ViCo测试，原论文自身FD/MSE/SID/Var；其RLHG/L2L/DIM行源于DIM | 区分原INFP与AvatarForcing/DyStream的INFP*；不能引用其他重实现值冒充原INFP |
| INFP interactive结果 | INFP Table1，p8 | 私有DyConv test；DIM在DyConv重训 | 取不到DyConv精确test不能加入统一新表；官方demo只能作定性材料 |
| ARIG listener结果 | ARIG Table2，p7 | ViCo；baseline源于DIM/INFP；作者视频反提3DMM；原文p7记70维 | 与后续ECHO将其标为76维存在表述差异，须保留冲突，不能替作者消解；不可按同维自动排名 |
| DiTaiListener | Table2/4 ViCo；Table3/5 RealTalk，p5、8 | 图像评价与EMOCA动作评价分开；INFP†直接取INFP；文本control来自listener观察 | test文本条件的可得性、实际baseline训练来源；无文本消融用于条件公平分析 |
| Avatar Forcing与其INFP* | Avatar Forcing Table1/3/4，p6、附录p14 | RealTalk/HDTF50/ViCo分开；motion为SPECTRE56；ViCo旧baseline引用DIM | 只引用其INFP*时注明FLOAT运动空间重实现；不要同时把它与原INFP合并成一行 |
| DyStream与其INFP* | DyStream Table1/2，p5–6 | speaking HDTF100及正文内部100；listener RealTalk38；MediaPipe；SID k40/20；主表多为offline | 不混用DIM的EMOCA表示与统计；2026-11-08核对DIM发布代码，SID同样使用k40/20，此前k15/9的描述不正确；核对online分组；原INFP*版本属于DyStream作者 |
| StreamAvatar | ECHO Fig5仅定性；要引数值须另取[StreamAvatar原文](https://arxiv.org/html/2512.22065v1)Table1或Table4 | 原生双方已知audio＋reference→全身视频；Table1自建short/long，Table4为EMTD110；FID/FVD/Q-Align/Sync/HKV/HA | 没有ECHO同协议下的StreamAvatar定量行；不能从demo推出ViCo指标，也不能误用同名DyStream蒸馏仓库 |
| Per-FLOAT | ECHO Fig3栅格原图，p11 | 仅定性比较，无实现定义；图内引用编号与当前文献表不一致 | 保留原标签与身份未明确；没有可直接引用数值，不能改名为Avatar Forcing或原FLOAT |
| ListenFormer | 原文Table1/2 ViCo，Table3 L2L，p5–6 | ViCo70维+图像，L2L56维动作；MFR直接引用；DSPN为比较方法 | 不把L2L行移入ViCo；未明确checkpoint与人物范围不擅自补写 |
| DSPN | 本次可核实的性能来源为ListenFormer Table1/2/3；方法论文DOI 10.1145/3607541.3616820 | 明确区分ViCo与L2L；引用复现需同时引用DSPN方法与ListenFormer结果来源 | 本次未取得DSPN原始全文表，不冒充已读其原表；ListenFormer复现就是这里所引数值的来源，不能改称DSPN原checkpoint |
| ReactFace | 作者替代版Table2，p10–13 | REACT2023 NOXI/RECOLA、806 test、30s；FaceVerse58；多个适当反应邻居；FRDvs/Var/Div/Rea/Corr/Syn | appropriateness邻居、采样数、DTW/TLCC和渲染；和单GT整段类别F1属于不同评价问题 |
| ReactFace额外RLD结果 | 作者替代版Table9，p15，说明见p12/15 | RLD即ViCo，按RLHG原3DMM体系；无appropriate邻居标签，Larec换为普通Lrec；FRRea/Var/Div/Syn，不报FRCorr | 这是单独训练／评测设置，不能沿主REACT表FaceVerse58解释；具体test清单、其他行checkpoint仍未披露 |
| OmniResponse | 原文Table2，p7–8 | 官方ResponseNet完整长对话；文本/音频/视频三组、时序自回归；UTMOS/LSE-D适用于相应有声窗口 | 官方testID和全对话条件；本地586 J pair/任何200子集不等于原协议；短声UTMOS N/A不能填0 |
| SpeakerVid-5M生成baseline | 原文Table2最终Dyadic设置，p9 | VidChatBench500未见speaker；条件setting和dyadic setting不同；CLIPdialog有五候选排序；SIM-o有reference voice | testset与候选、预处理、sync、64维情绪FID；处理工具权重不能代替生成模型 |
| OmniPersona | 本地在审论文Table2/3/4，p7–8，附录p15–20 | VidChatBench、原OmniResponse、自选Seamless60条三种交互状态约50s；历史与声音身份条件；EA/AVS/OQ人工协议 | 与本文当前短音频设定不同；精确60条ID及baseline运行来源未披露；附录同文本控制仍不是同generator控制 |
| 独立OVI及OmniPersona附加比较 | OmniPersona附录Table13/15/16，p18–20 | 同SpeakerVid预测response text控制；自动筛选20条anger；5s/125帧448px单H200完整pipeline速度。Table16的OVI与Qwen＋OVI主表行不同 | 不把速度表当质量对比；同文本不等于共享训练，自动anger不是人工标签。OVI组件可取不等于OmniPersona训练权重可取 |
| ECHO | 原文Table1–4，p12–14 | RealTalk IHG、HDTF50、ViCo；EMOCA50+6；人工E-Appro和Qwen3-Omni E-Consis | 多个旧数值与前表一致但无统一引用声明；不要把“看起来一样”升级为确认引用。Arig维数冲突单列 |
| OmniHuman-1.5及其动画比较方法 | 原文Table3/4，p10–12 | 自建57人多主体；CelebV-HQ100、CyberHost269；给定目标audio，IQA/ASE/Sync/FID/FVD/HKC/HKV | 语音驱动动画不是自主声音反馈，匿名商业系统无法定位，不能引用到listener主表 |

## 当前框架下可以先固定的表格组织原则

1. **共同输入主表**：当前speaker audio＋listener首帧，固定自己测试清单与同一外部评测。联合生成系统比较视觉、声音、语境、视听协调；级联Qwen→OVI组件有官方资源，整体适配尚未实测。OVI1.1单次至多10s，长pair须统一分段/时长协议。GLARE p16另有固定Tref与previous motion，其来源和初始化必须披露后再谈严格共同输入。
2. **视觉listener表**：不生成声音的方法只评视觉和可适用人工项目。可以再分audio-only与speaker-audio/video额外输入。AvatarForcing作者已确认user视觉可置0，是适配线索，不是已经验收成功。
3. **已知target音轨上限／补充表**：INFP、DyStream、ARIG、ECHO等原生有双方音轨的系统，如果额外提供真实listener音轨，必须单列；如果self置0，则它们不能生成声音，声音项N/A。不要把目标语音当未知输入给本文、却给基线当真值。
4. **文献参考表**：保留上述各原始协议及方法来源。无法对齐原测试片段／训练适配／表示／采样时，只报告参考，不和我们的新指标混算增益。

当前benchmark三组评价已由作者确认，但类别表、检测器、阈值、人工量表和最终测试名单尚未冻结；本调研不擅自把候选协议变成已经执行的实验。
