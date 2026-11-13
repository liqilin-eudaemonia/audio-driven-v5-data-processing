# 官方资源与issue补充核查（2026-11-07）

本页记录新增核查结果；完整逐项资源状态见[85个名称的资源矩阵](RESOURCE_MATRIX.md)，实验安排见[建议矩阵](EXPERIMENT_RECOMMENDATION.md)。下载入口存在、文件可以下载、模型成功推理、原论文协议复现成功是不同状态；本次调研没有启动训练或GPU推理。

| 方法 | 已检查的官方入口／issue | 核实结果 | 对本文的用途 |
|---|---|---|---|
| Avatar Forcing | [README](https://github.com/TaekyungKi/AvatarForcing)、[download_weights.sh](https://github.com/TaekyungKi/AvatarForcing/blob/main/download_weights.sh)、[#5](https://github.com/TaekyungKi/AvatarForcing/issues/5)、[#11](https://github.com/TaekyungKi/AvatarForcing/issues/11)、[#13](https://github.com/TaekyungKi/AvatarForcing/issues/13) | 官方有核心模型与motion AE下载入口。**脚本实际用两个Google Drive文件，HuggingFace命令仅下载wav2vec2**；README笼统说HF下载不能代替检查脚本。#5作者明确可把user视觉条件置0，视觉输入并非必须。#11的解释来自其他用户liuandian，不能当作者承诺。#13问题本身不能证明解决。 | 可增加audio-only条件的视觉listener比较；target自身音频置零、user audio有效，且在代码层核验分支。它不生成声音，音频结果N/A。核心下载payload与推理仍待验收。 |
| DualTalk | [官方README](https://github.com/ZiqiaoPeng/DualTalk)、[HF模型](https://huggingface.co/ZiqiaoPeng/DualTalk)、[#7](https://github.com/ZiqiaoPeng/DualTalk/issues/7)、[#10](https://github.com/ZiqiaoPeng/DualTalk/issues/10) | 官方给出HF核心模型、demo/train/test/render/metric入口，demo需要双方音频和一方blendshape。#10用户指出公开数据只有音频和3D mesh，没有原视频。#7的SID问题已进一步核实[当前代码](https://github.com/ZiqiaoPeng/DualTalk/blob/main/metric/metric.py)：函数`calcuate_sid(gt,pred,k)`拟合首参数，调用却传入`pre_exp,gt_exp,40`（jaw/pose同理），因此当前代码方向颠倒。**这不证明发表数字全部使用此代码或均错误**。HF目录实际列出`DualTalk.pth` 2.59GB；本次未下载完整payload。 | 新增可运行候选，属于3D交互动作比较，不能冒充真实人物视频输出；若要统一图像分数，需单独renderer和说明。双音轨及partner动作条件也需披露。 |
| DyStream | [官方项目](https://robinwitch.github.io/DyStream-Page/)、[代码](https://github.com/RobinWitch/DyStream)、[HF权重](https://huggingface.co/robinwitch/DyStream) | 有模型下载和推理。README的“speaker audio only”示例把audio写入self，实际是说话模式；不能直接用于listener比较。本地此前已明确speaker进入other、self零音轨。 | 已有本地ViCo43成功推理证据，需复核最新版本；适配后的视觉baseline不产生listener音频。 |
| PCH | [论文链接现重定向到hzwer官方仓库](https://github.com/hzwer/MM2022-ViCoPerceptualHeadGeneration) | 公开的是替换ViCo baseline的局部模块和U2Net背景处理；README的U2Net下载仅分割权重，不是driver。历史[#2](https://github.com/hzwer/MM2022-ViCoPerceptualHeadGeneration/issues/2)作者没有提供核心driver，转荐其他实现。 | 可训练/可重实现，不能登记为现成完整checkpoint。文献引用其challenge成绩或MFR/Custom等目标数据重实现成绩时须逐表注明。 |
| SpeakerVid-5M | [代码](https://github.com/Dorniwang/SpeakerVid-5M-Code)、[资源目录](https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models)、[#2](https://github.com/Dorniwang/SpeakerVid-5M-Code/issues/2) | README明确开放数据标注、收集、处理；benchmark给500条testset与metrics，下载assets/speaker/weights是处理和评分资源。不能按文件夹名“speaker ckpt”推断是论文AR对话生成器。历史issue#10中第三方把这些称为baseline checkpoints不作为作者生成器发布证据。 | 可用其评测框架与原始VidChatBench，生成器代码/权重仍未证实可用；需要引用生成结果时用SpeakerVid原文Table2的正确设置。 |
| OmniResponse | [官方仓库](https://github.com/awakening-ai/OmniResponse) | 当前TODO中推理、renderer/pretrained、各LLM checkpoint、训练均未勾选；不能用已公开ResponseNet数据证明生成模型可推理。 | 不能在自切J测试集直接引用原文长对话分数；仅作原协议文献表参考或等待模型发布。 |
| OmniHuman-1.5 | [官方项目](https://omnihuman-lab.github.io/v1_5/) | 目前核查页给论文与展示，没有提供核心模型下载入口；尚不能扩展为“全网不存在权重”的断言。 | 给定目标语音的动画背景，不与自主listener音频生成混表。引用原文Table4须对齐CelebV-HQ100/CyberHost269，而非ViCo或SeamlessJ。 |

## 本轮进一步收齐的证据

- LFI的[官方README](https://github.com/jonepatr/lets_face_it)给出Glow训练命令、特征提取和FLAME可视化，归为可重训，未核实核心checkpoint下载。
- ECHO已找到[官方仓库](https://github.com/xk0720/ECHO)，当前文件树只有LICENSE和README；修正“未找到官方入口”，仍不能列为已有可执行实现。
- DiTaiListener的[作者可访问页](https://cv.maxi.su/DiTaiListener/)中Code/Model为占位，与旧项目HTML空href相符。ihp-lab访问失败本身不作未发布证据。
- [InterActHuman官方页](https://zhenzhiwang.github.io/interacthuman/)明确示例代码不是论文原实现，且不能提供checkpoint。不要把Example Code登记为完整开源复现。
- [GLARE官方eval](https://github.com/lzk901372/glare/tree/main/eval)已逐项检查reaction输入契约：六类、tIoU≥0.5一对一匹配、已平滑CSV、R-ATD报告×100。这不是完整检测器，也不是本文整段类别出现F1。
- 外围13个说话动画方法、3个mesh方法、Moshi/F5-TTS、Qwen/DeepSeek、Qwen→OVI及独立OVI已分别登记官方入口。没有下载所有大模型或声称原表一键复现；商业LLM历史API版本也未验收。
- DIM今日README仍无核心下载，issue2只有用户请求；与旧commit协议审计结合保留“可重训、原表未严格对齐”。L2L仍明确不提供写实renderer，四人物motion权重与统一视频基线分开。

## 仍须保留的边界

已补查图中／正文比较与官方文字补充，完整编号表清单见TABLE_CENSUS，逐方法来源见ROW_PROVENANCE。作者未披露的checkpoint、重训方式、精确test或prompt，调研结论就是“未披露”，不能用数字一致、仓库存在或依赖权重替作者补写。文件目录存在与payload成功下载、模型本地运行和原协议复现仍是独立层级。

本轮另确认：StreamAvatar原项目仅为展示页，ECHO只定性比较；ECHO栅格Fig3另有Per-FLOAT未明确变体。GLARE p16的固定Tref视频条件须披露，不能只记audio＋首帧。DiffListener补充A.3明确RLHG/PCH去emotion后重训。DIM issue3的models/lib缺失已有作者在9fc92f7修复，不属于当前仍缺代码的证据。
