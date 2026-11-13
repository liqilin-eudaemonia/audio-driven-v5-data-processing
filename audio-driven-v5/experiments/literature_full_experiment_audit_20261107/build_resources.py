"""Curated resource evidence, distinct from inference/protocol acceptance.

URLs are official entry points checked during this audit. A published download
link is not a completed payload download, nor a successful local reproduction.
Unknowns are deliberate findings, not promises about future releases.
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
R = {}


def add(label, decision, code, weights, metrics, fit, sources, limitation):
    assert label not in R, label
    R[label] = dict(resource_status=decision, code=code, core_weights=weights,
                    evaluation=metrics, applicability=fit, sources=sources,
                    limitation=limitation, checked_date='2026-11-07',
                    new_local_inference_in_this_audit=False)


add('RLHG', 'trainable_no_verified_core_download',
    '官方训练/预测与PIRender适配代码', '未在当前README定位listener核心权重；renderer下载不是driver',
    '官方evaluation图像指标；动作距离须保留L1×100定义',
    '视觉listener；原生还需speaker motion和attitude',
    ['https://github.com/dc3ea9f/vico_challenge_baseline',
     'https://github.com/dc3ea9f/vico_challenge_baseline/issues/1',
     'https://github.com/dc3ea9f/vico_challenge_baseline/issues/24'],
    'ViCo项目集与challenge集有差异；issue24是ArcFace依赖下载，不是RLHG核心模型发布。')
add('PCH', 'partial_code_reimplementation_needed',
    '官方为ViCo替换模块与背景融合，非完整独立系统', 'U2Net是分割模型；未证实PCH核心driver可取',
    '沿ViCo；challenge表是各队提交', '视觉listener；需重新整合driver/renderer并训练',
    ['https://github.com/hzwer/MM2022-ViCoPerceptualHeadGeneration',
     'https://github.com/hzwer/MM2022-ViCoPerceptualHeadGeneration/issues/2'],
    '作者issue2明确建议使用其他更完整系统；不能仅记为一条命令可重训。')
add('L2L', 'official_weights_motion_only',
    '官方train/test，人物数据与四人物motion模型', 'Conan/Fallon/Stephen/Trevor，官方unpack_models.sh含Drive入口',
    '有test生成motion；没有已核实的一键全主表评分套件',
    '动作listener补充；原生speaker audio+motion；统一视频表另需renderer',
    ['https://github.com/evonneng/learning2listen',
     'https://github.com/evonneng/learning2listen/blob/main/src/scripts/unpack_models.sh',
     'https://github.com/evonneng/learning2listen/issues'],
    'README明确不提供写实可视化代码。本地历史四模型推理证据不等于目标数据重训；不恢复或覆盖其他训练任务。')
add('MFR-Net', 'no_verified_official_implementation',
    '未定位可核实的官方实现', '未定位官方核心下载', '未定位独立官方评分套件；论文指标可参照ViCo',
    '目前文献参考；可另行按论文重实现',
    ['https://arxiv.org/abs/2308.16635',
     'https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_CustomListener_Text-guided_Responsive_Interaction_for_User-friendly_Listening_Head_Generation_CVPR_2024_paper.pdf'],
    'CustomListener正文明确当时ELP/MFR代码不可用；本次精确标题及GitHub检索仍未定位，不声称全网绝对没有。')
add('ELP', 'no_verified_official_implementation',
    '未定位可核实的原版实现；DIM的ELP是作者重实现', '未定位原版核心下载',
    '原表FD命名与公式冲突；DIM重实现用另一动作协议', '原版文献参考；DIM重实现不可冒名原版',
    ['https://arxiv.org/abs/2310.00068',
     'https://openaccess.thecvf.com/content/ICCV2023/papers/Song_Emotional_Listener_Portrait_Neural_Listener_Head_Generation_with_Emotion_ICCV_2023_paper.pdf'],
    '精确标题/GitHub检索未定位官方模型。原ViCo100维体系与DIM EMOCA56不可拼表。')
add('CustomListener', 'no_verified_official_implementation',
    '已取得官方论文和补充，未定位可执行官方模型仓库', '未定位官方核心下载',
    '论文/补充给指标；未定位独立官方执行套件', '文本控制视觉listener，需注明推理文本来源',
    ['https://openaccess.thecvf.com/content/CVPR2024/html/Liu_CustomListener_Text-guided_Responsive_Interaction_for_User-friendly_Listening_Head_Generation_CVPR_2024_paper.html',
     'https://openaccess.thecvf.com/content/CVPR2024/supplemental/Liu_CustomListener_Text-guided_Responsive_CVPR_2024_supplemental.pdf'],
    '不能把DiffListener描述其某一比较设置为GT文本，推广成CustomListener所有推理均用GT。')
add('DIM', 'trainable_no_verified_core_download',
    '官方VQ/pretrain/finetune/test/renderer', 'README无核心下载；issue2仍为用户请求且未见作者提供',
    'mymetrics/metrics和test；已审计默认speaker常量与10次GT择优',
    '重训后可做motion/视觉listener；原生speaker motion额外条件',
    ['https://github.com/Boese0601/Dyadic-Interaction-Modeling',
     'https://github.com/Boese0601/Dyadic-Interaction-Modeling/issues/2',
     'https://github.com/Boese0601/Dyadic-Interaction-Modeling/issues/3'],
    '公开test路径不证明原主表协议；需固定有效test、输入与择优规则。旧commit审计见dim_protocol/AUDIT.json。')
add('ReactFace', 'trainable_no_verified_core_download',
    '官方train/inference/evaluate/render/metric', '核心checkpoint请求issue1未见答复；FaceVerse/PIRender是依赖',
    '官方多适当反应邻居/DTW/TLCC/FVD等评测入口',
    '重训视觉listener；原生speaker AV，FaceVerse58',
    ['https://github.com/lingjivoo/ReactFace', 'https://github.com/lingjivoo/ReactFace/issues/1'],
    '官方GT与邻居下载入口不等于核心生成器已发布；单GT事件F1不等于其appropriateness协议。')
add('DiffListener', 'project_only_no_generator',
    '所核官方仓库为项目展示网站', '未见核心模型链接', '未见独立官方执行入口',
    '目前文献参考/另行重实现', ['https://github.com/nuo1wang/DiffListener'],
    '仓库static/index与网站模板README不能登记为生成代码，相关补充已在旧审计取得。')
add('ListenFormer', 'trainable_historical_weight_link_unavailable',
    '官方run.sh/eval.sh，沿ViCo渲染/评分', 'issue1作者提供过百度链接；用户此前确认过期，无替代入口证据',
    '沿ViCo官方evaluation；L2L表人物/权重范围未披露', '可重训视觉listener，当前不能承诺直接下载推理',
    ['https://github.com/liushenme/ListenFormer', 'https://github.com/liushenme/ListenFormer/issues/1'],
    'issue关闭只表示历史回复，不证明今天可下载；百度payload本次未重新取出。')
add('DiTaiListener', 'project_placeholders_no_generator',
    '作者项目页Code为占位项；未定位实现', 'Model为占位项', '论文含指标，未见已发布独立评分入口',
    '文本可控视觉listener，目前文献参考',
    ['https://cv.maxi.su/DiTaiListener/',
     'https://havent-invented.github.io/DiTaiListener/', 'https://arxiv.org/abs/2504.04010'],
    'ihp-lab入口访问失败不作未发布证据；可访问作者页的Code/Model占位才是所见状态。')
add('GLARE', 'trainable_no_verified_core_download',
    '官方train/gen和reaction eval', 'FLOAT初始化已列；GLARE训练完成核心权重请求issue1未答',
    '官方eval含R-F1/R-tIoU/R-ATD/R-FID；需先准备已平滑CSV',
    '任务接近的视觉listener；但p16另给固定Tref视频和previous motion，不能直接称严格共同输入',
    ['https://github.com/lzk901372/glare', 'https://github.com/lzk901372/glare/issues/1',
     'https://github.com/lzk901372/glare/issues/2',
     'https://github.com/lzk901372/glare/blob/main/eval/README.md'],
    '精确源片段/角色/裁框与固定Tref来源未补。官方tIoU≥0.5事件匹配与本文整段出现F1不同；R-ATD实现另×100。')
add('DSPN', 'no_verified_official_implementation',
    '未定位官方代码', '未定位核心权重', 'ListenFormer提供作者复现的指标，非DSPN原始工具',
    '补全方法并集；目前引用ListenFormer复现并注明身份',
    ['https://doi.org/10.1145/3607541.3616820', 'https://www.sigmm.org/opentoc/MCGE23',
     'https://github.com/liushenme/ListenFormer'],
    '精确全名Responsive Listening Head Synthesis with 3DMM and Dual-Stream Prediction Network；未取得原全文的表格来源仍待补，不猜原始数值。')
add('LFI', 'trainable_no_verified_core_download',
    '官方Glow训练、特征提取与FLAME可视化', '未在README定位核心生成器下载；FLAME是模型依赖',
    '官方训练/测试路径；L2L重训比较另有原论文协议', '动作生成，可重训；无原始音视频的LFI数据不能做图像或音频表',
    ['https://github.com/jonepatr/lets_face_it', 'https://jonepatr.github.io/lets_face_it/',
     'https://github.com/jonepatr/lets_face_it/issues/1', 'https://github.com/jonepatr/lets_face_it/issues/4'],
    '不能列成只有文献可用；issue1/4涉及数据文件/目录适配，本次未实际重训。L2L引用其原split重训是motion-only，另在L2L自身数据重训含更多条件。')
add('INFP', 'project_only_no_generator',
    '官方仓库为homepage', '未见官方核心可下载权重', '原文指标；没有已验证官方可执行套件',
    '双方已知音轨→视频；现阶段原协议文献参考', ['https://github.com/grisoon/INFP'],
    'AvatarForcing/DyStream INFP*是两套不同重实现，不能当原INFP模型下载。')
add('ARIG', 'project_only_no_generator',
    '项目页Paper/Code均回到同页，未定位实现', '未见核心下载', '原文指标，未定位执行入口',
    '额外目标audio和user AV条件的交互视频参考', ['https://jinyugy21.github.io/ARIG/'],
    '标题、作者与代码链接须核对；原文70维与ECHO标76维冲突保持记录。')
add('Avatar-Forcing', 'official_core_weight_links_adaptation_needed',
    '官方inference及相关模块', 'download_weights.sh用Drive下载flow_transformer与motion_autoencoder',
    '可用通用外部评测；精确原表清单/完整一键评分尚未验收',
    '作者issue5确认user视觉可置零；self audio置零后是视觉listener适配候选',
    ['https://github.com/TaekyungKi/AvatarForcing',
     'https://github.com/TaekyungKi/AvatarForcing/blob/main/download_weights.sh',
     'https://github.com/TaekyungKi/AvatarForcing/issues/5',
     'https://github.com/TaekyungKi/AvatarForcing/issues/11',
     'https://github.com/TaekyungKi/AvatarForcing/issues/13'],
    '有官方核心链接≠本地已跑通；HF wav2vec不是核心。#11是第三方答复；置零适配须实际代码验证。')
add('ECHO', 'official_repository_placeholder',
    '已定位官方xk0720/ECHO，但目前只有LICENSE/README', '没有核心下载链接', '未见执行入口',
    '交互视频/情绪语境方法，目前文献参考',
    ['https://github.com/xk0720/ECHO', 'https://xk0720.github.io/ECHO/',
     'https://github.com/xk0720/ECHO/issues'],
    '修正此前未找到官方仓库的状态；仓库自称implementation不等于代码已经提交。')
add('StreamAvatar', 'project_only_no_generator',
    '所核官方StreamAvatar组织仓库为展示网站，未见原实现', '官方项目页未定位原版核心checkpoint',
    'ECHO Fig5仅定性；若引用数值需另取StreamAvatar原表协议',
    '双方已知音轨＋首帧→交互全身视频，不生成回应声音',
    ['https://streamavatar.github.io/', 'https://github.com/StreamAvatar/streamavatar.github.io',
     'https://arxiv.org/abs/2512.22065', 'https://arxiv.org/html/2512.22065v1'],
    'CXP-2024/DTG777同名仓库是DyStream/AROD蒸馏，不是ZhiyaoSun等人的论文原实现；不把这些权重误配。原项目网站本身也不等于生成代码发布。')
add('DyStream', 'official_weights_historical_local_inference',
    '官方offline/online inference与模型代码', 'HF robinwitch/DyStream核心模型目录',
    '论文MediaPipe协议；统一外部视觉指标可另跑',
    'speaker输入other，self零音轨做listener视觉适配；不自主生成音频',
    ['https://github.com/RobinWitch/DyStream', 'https://huggingface.co/robinwitch/DyStream'],
    '本地历史ViCo43已跑；README speaker-only示例self含speaker音频，不可直接当listener；主表offline与online分开。')
add('DualTalk', 'official_core_weights_extra_conditions',
    '官方train/test/demo/render/metric', 'HF ZiqiaoPeng/DualTalk.pth，目录显示2.59GB',
    'metric.py可取；SID调用已核实与函数参数顺序颠倒',
    '双音轨+partner blendshape→3D动作；额外输入补充表，写实视频另需renderer',
    ['https://github.com/ZiqiaoPeng/DualTalk', 'https://huggingface.co/ZiqiaoPeng/DualTalk/tree/main',
     'https://github.com/ZiqiaoPeng/DualTalk/blob/main/metric/metric.py',
     'https://github.com/ZiqiaoPeng/DualTalk/issues/7', 'https://github.com/ZiqiaoPeng/DualTalk/issues/10'],
    '当前函数(gt,pred,k)拟合首参数，实际传(pre,gt,k)，需修复后评分；不能据此断言发表数字均错误。公开数据不等于原写实视频。')
add('SpeakerVid-5M', 'curation_and_metrics_not_generator',
    '官方数据收集/注释/筛选代码及benchmark', 'data-curation-models为处理依赖，未证实AR生成器核心发布',
    'VidChatBench500、图像/情绪/Sync/SIM-o等工具；CLIPdialog依赖候选文本',
    '联合AV直接相关，但当前无法按已公开资源承诺实测生成器',
    ['https://github.com/Dorniwang/SpeakerVid-5M-Code',
     'https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models',
     'https://github.com/Dorniwang/SpeakerVid-5M-Code/issues/2'],
    'weights/speaker是声纹依赖；第三方称baseline checkpoint不是作者生成模型发布证据。')
add('OmniResponse', 'dataset_only_generator_todo',
    '官方README推理/训练/renderer仍TODO', '各LLM核心checkpoint仍TODO',
    '官方评测章节占位；本地evaluation是另行整合的工具，不能冒充作者原表脚本',
    '联合AV直接相关，目前原长对话协议文献参考',
    ['https://github.com/awakening-ai/OmniResponse', 'https://omniresponse.github.io/'],
    'ResponseNet开放≠生成模型开放；本地586/200短pair不是原长对话协议。')
add('OmniPersona', 'anonymous_manuscript_no_verified_release',
    '本地ICLR2027匿名在审稿，未核实对应公开实现', 'OVI基础模型可取不等于OmniPersona训练权重',
    '论文列协议，精确Seamless60清单与实现未公开核实',
    '联合AV但原生历史/声纹条件更多，目前文献参考', [],
    '只按本地完整题名Persona-Embodied Agents for Multimodal Conversational Interaction识别；不把同名个性化图像编辑项目误配，也不推测匿名作者。')

# Peripheral methods are recorded individually to preserve the full comparison union.
ANIMATION = [
    ('SadTalker', 'OpenTalker/SadTalker', 'GitHub Releases/Google Drive；256/512 safetensors'),
    ('AniTalker', 'X-LANCE/AniTalker', 'HF taocode/anitalker_ckpts；stage1与stage2；额外训练代码未完整公开'),
    ('EchoMimic', 'antgroup/echomimic', 'HF BadToBest/EchoMimic核心UNet/motion_module'),
    ('Hallo', 'fudan-generative-vision/hallo', 'HF fudan-generative-ai/hallo'),
    ('Hallo3', 'fudan-generative-vision/hallo3', 'HF fudan-generative-ai/hallo3与CogVideoX依赖'),
    ('FLOAT', 'deepbrainai-research/float', '官方download_checkpoints.sh/Google Drive float.pth'),
    ('Sonic', 'jixiaozhong/Sonic', 'HF LeonJoe13/Sonic及SVD依赖'),
    ('Real3DPortrait', 'yerfor/Real3DPortrait', '官方Drive/Baidu模型，训练及推理代码'),
    ('SkyReels-A1', 'SkyworkAI/SkyReels-A1', 'HF Skywork/SkyReels-A1；音频驱动依赖额外motion prior'),
    ('FantasyTalking', 'Fantasy-AMAP/fantasy-talking', 'HF acvlab/FantasyTalking与Wan依赖'),
    ('OmniAvatar', 'Omni-Avatar/OmniAvatar', 'HF OmniAvatar/OmniAvatar-14B或1.3B，与Wan依赖'),
    ('MultiTalk', 'MeiGen-AI/MultiTalk', 'HF MeiGen-AI/MeiGen-MultiTalk，与Wan依赖'),
    ('LiveAvatar', 'Alibaba-Quark/LiveAvatar', 'HF Quark-Vision/Live-Avatar LoRA与Wan2.2-S2V-14B'),
]
for name, repo, weight in ANIMATION:
    add(name, 'official_animation_weight_links', '官方动画推理代码', weight,
        '可用共同外部视频评测；不据此声称原引用论文的test或全部指标已可重现',
        '给定要说的目标音频→动画；适合作为renderer/已知目标音频补充，不是自主listener回应',
        ['https://github.com/' + repo], '核心入口已核；未下载所有大模型或逐一运行。原论文采用的具体版本未披露时保持未知。')
for name, url in [('OmniHuman-1', 'https://omnihuman-lab.github.io/'),
                  ('OmniHuman-1.5', 'https://omnihuman-lab.github.io/v1_5/'),
                  ('Loopy', 'https://loopyavatar.github.io/')]:
    add(name, 'project_only_no_verified_core_download', '官方展示页，未找到核心实现链接',
        '未核实官方核心checkpoint', '论文指标；未核实原协议一键工具',
        '给定目标音频动画背景，当前文献参考', [url],
        '仅结论为所核官方页未提供，不将同名其他项目或第三方商业接口当原版官方模型。')
add('InterActHuman', 'example_not_original_no_checkpoints',
    '官方明确Example Code不是论文原实现', '官方明确不能提供预训练checkpoint',
    '提到Phantom数据/工具，不等于作者模型可跑', '多主体已知audio动画，文献参考',
    ['https://zhenzhiwang.github.io/interacthuman/'], '按官方明确免责声明判定，不能列为已开源完整实现。')

for name, repo, weight in [('MeshTalk', 'facebookresearch/meshtalk', 'GitHub pretrained_models_v1.0 release'),
                           ('FaceFormer', 'EvelynFan/FaceFormer', '官方Drive biwi.pth/vocaset.pth'),
                           ('CodeTalker', 'Doubiiu/CodeTalker', '官方Drive两阶段BIWI/VOCaset模型')]:
    add(name, 'official_3d_speech_weights', '官方train/test', weight,
        '原生mesh指标；DIM Table4重训到5023顶点后协议不同',
        '自身语音→3D说话动作，非listener自主反应', ['https://github.com/' + repo],
        '不能拿原生拓扑的官方模型数字直接替代DIM重训BIWI结果；MeshTalk示例训练loader为dummy数据，实际训练需适配。')
add('DIM-Speaker', 'variant_of_DIM_trainable', 'DIM speaker分支训练/测试代码', '同DIM核心权重缺口',
    'DIM BiWi Table4 LVE/FDD', '说话动作补充，不是listener方法',
    ['https://github.com/Boese0601/Dyadic-Interaction-Modeling'], '见DIM与原论文附录5023顶点转换。')
add('Moshi', 'official_audio_weights', '官方推理', 'kyutai官方HF moshi-v0.1模型',
    '音频/文本可另评，原OmniResponse分数仍需其prompt/时序协议', '音频对话，无视觉输出',
    ['https://github.com/kyutai-labs/moshi'], '不能直接当联合AV系统；完整回复与短backchannel需匹配任务指令。')
add('F5-TTS', 'official_tts_weights', '官方训练/推理', 'HF SWivid/F5-TTS',
    'TTS质量/声纹/适用同步指标', '文本+参考音频→声音；回复内容与时刻需上游决定',
    ['https://github.com/SWivid/F5-TTS', 'https://huggingface.co/SWivid/F5-TTS'],
    '给GT文本或GT音频的TTS比较须单列，不能说它自行理解speaker并选择反馈。')
for name, url in [('Qwen-7B-Chat', 'https://huggingface.co/Qwen/Qwen-7B-Chat'),
                  ('DeepSeek-R1', 'https://huggingface.co/deepseek-ai/DeepSeek-R1-0528')]:
    add(name, 'official_language_weights', '官方文本推理', '官方模型仓库',
        'OmniResponse文本协议，非独立AV评测', '可作文本规划组件，需ASR与生成器组成pipeline', [url],
        'OmniResponse p8明确DeepSeek版本2025-05-28，故应核R1-0528而非泛用初版R1；提示/采样未逐项披露。Qwen沿其原表名称；不等价OmniResponse内训练模型。')
for name in ['GPT-4o', 'GPT-4', 'GPT-o1', 'Claude-Sonnet-4', 'Gemini-2.5-Flash']:
    add(name, 'historical_commercial_language_baseline', '原论文使用的商业文本模型调用',
        '不是公开下载模型权重', '原OmniResponse Table2文本指标；本次未验证历史API仍可访问',
        '语言侧参考，不是可直接运行的联合AV baseline', [],
        'OmniResponse p8明确Claude版本2025-05-14；其他未给的snapshot/prompt不猜补。若重跑还需核对历史API可用性。')
add('Qwen2.5-Omni+OVI', 'open_components_cascade_not_yet_benchmarked',
    'Qwen与Ovi各有官方推理，级联需适配', 'Qwen/Qwen2.5-Omni-7B；chetwinlow1/Ovi',
    '共同外部AV评测；不是OmniPersona的已发布完整baseline脚本',
    '可构造当前audio+portrait共同输入的联合AV级联对照',
    ['https://github.com/QwenLM/Qwen2.5-Omni', 'https://huggingface.co/Qwen/Qwen2.5-Omni-7B',
     'https://github.com/character-ai/Ovi', 'https://huggingface.co/chetwinlow1/Ovi'],
    '需固定回复长短/时刻/静默/提示策略；不给GT caption或GT事件；OVI本身不理解incoming audio。')
add('OVI', 'official_joint_av_backbone_weights',
    '官方joint AV推理及download_weights.py', '官方HF chetwinlow1/Ovi，含1.1版本',
    '可用共同AV外部指标；OmniPersona Table16是速度表非listener质量表',
    '图像+文本→联合AV，可作本文无条件适配的基础模型对照；不能直接理解speaker audio',
    ['https://github.com/character-ai/Ovi', 'https://huggingface.co/chetwinlow1/Ovi'],
    '单独OVI与Qwen2.5-Omni+OVI是不同实验行；分别登记，不能遗漏也不能声称OVI自身完成speaker理解。')
add('Per-FLOAT', 'unresolved_figure_variant_no_verified_checkpoint',
    'ECHO Fig3图片中的真实标签；主文未解释其具体实现', '未核实该变体核心权重；不能拿FLOAT主模型替代',
    '仅定性行，无定量指标可引用', '保留原图身份，不擅自合并到Avatar Forcing或FLOAT原版',
    [], '图内引用序号[51]不对应当前文献的FLOAT，DIM/DualTalk图内编号亦不对应当前文献表；按版本冲突记录。')

VARIANT_PARENT = {
    'OmniResponse-LSTM': 'OmniResponse', 'OmniResponse-AV-LLM': 'OmniResponse',
    'INFP-reimpl-AvatarForcing': 'Avatar-Forcing', 'INFP-reimpl-DyStream': 'DyStream',
    'INFP-I': 'DyStream', 'L2L-CANDOR': 'DIM', 'L2L-Random-Walk': 'L2L',
}
for name, parent in VARIANT_PARENT.items():
    p = R[parent]
    add(name, 'paper_specific_variant_resources_not_independent_release',
        '属于' + parent + '论文中的专门对照，父仓库存在不证明该变体已发布',
        '未独立核实该表格变体checkpoint', '按其出现论文的表号/数据/表示记录',
        '保留变体行身份，不合并为原版方法', p['sources'],
        '引用来源见METHOD_OCCURRENCES和PROTOCOL_MAP；不能借父方法主模型权重声称该行已复现。')


def main():
    registry = json.loads((BASE / 'METHOD_REGISTRY.json').read_text())
    for row in registry:
        name, group = row['label'], row['group']
        if group == 'long_video_generation_strategies':
            add(name, 'inference_strategy_not_independent_generator',
                'DiTaiListener作者在相同生成器上的长视频生成策略对照', '不需独立模型权重；依赖基准生成器',
                'ViCo>4s，CVF补充Table6 FlowConsistency/MHE/S3/NIQE',
                '长视频拼接/递推策略，不是新的完整listener生成器',
                ['https://openaccess.thecvf.com/content/ICCV2025/supplemental/Siniukov_DiTaiListener_Controllable_High_ICCV_2025_supplemental.pdf'],
                '保留论文使用的Teacher Forcing名称；其推理描述为上一段预测末帧递推，不擅自改称输入了GT未来帧。')
        elif group == 'controls':
            add(name, 'nonparametric_control_no_checkpoint_required',
                '按引用论文定义构造，部分作者仓库含实现', '不需要生成模型checkpoint',
                '对应论文同一指标/划分；检索只用训练库', '控制/检索对照，按任务适用范围选择', [],
                'Random/Median/Mirror等名称在不同论文的细节不一定相同，保留出现位置。')
        elif group == 'challenge_teams':
            add(name, 'leaderboard_submission_not_independent_public_model',
                'PCH Table1引用官方挑战赛队伍提交，未识别为独立公开方法',
                '不因团队名推断存在公开模型', 'ViCo challenge服务器协议',
                '完整保留被比较名单，通常无需作为本文额外模型逐个复现',
                ['https://vico-challenge.github.io/'], '若单独重现某队结果，需要该队技术报告/模型，不能用别的ViCo代码代替。')
    assert set(R) == {x['label'] for x in registry}, (set(R) ^ {x['label'] for x in registry})
    out = [dict(row, **R[row['label']]) for row in registry]
    (BASE / 'RESOURCE_MATRIX.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 全部方法资源与执行范围（2026-11-07）', '',
             f'本表逐项覆盖当前论文表格／图／正文登记的{len(out)}个名称；核心任务相关候选见对应三区，数量见METHOD_COUNTS。', '',
             '**有官方下载入口不等于本地推理通过；有评分代码不等于原表可复现。** 本次没有新增训练或GPU推理。原论文采用checkpoint/再训练方式未披露时仍保留未知；来源追溯与协议见 [PROTOCOL_MAP.md](PROTOCOL_MAP.md)，出现表格见 [METHOD_OCCURRENCES.json](METHOD_OCCURRENCES.json)。', '',
             '状态说明：`official_*`表示核实官方入口，`trainable_*`表示有训练路径但核心模型尚缺，`project_*`表示展示或占位；`no_verified_*`只表示在已记录的官方入口和精确检索中未核实，不能推导为全网绝无资源。', '']
    for group in dict.fromkeys(x['group'] for x in out):
        lines += ['## ' + group, '', '| 方法 | 代码／核心权重 | 评测与本文适用范围 | 证据／限制 |', '|---|---|---|---|']
        for row in out:
            if row['group'] != group:
                continue
            links = ' · '.join(f'[官方入口{i+1}]({s})' for i, s in enumerate(row['sources']))
            if not links:
                links = '本地PDF出现位置见METHOD_OCCURRENCES；未核实对应公开实现'
            cells = [row['label'], row['code'] + '；' + row['core_weights'],
                     row['evaluation'] + '；' + row['applicability'], links + '。' + row['limitation']]
            lines.append('| ' + ' | '.join(x.replace('|', '/') for x in cells) + ' |')
        lines.append('')
    (BASE / 'RESOURCE_MATRIX.md').write_text('\n'.join(lines) + '\n')
    counts = Counter(x['resource_status'] for x in out)
    print(json.dumps(dict(labels=len(out), decisions=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
