# ViCo参考方法：逐方法交接索引

更新：2026-11-11。飞书主表20个方法，加用户本轮要求的Avatar Forcing，共**21个方法、21份独立文档**。L2L四人物属于同一个方法；ViCo/RLHG、PCH/MM2022为别名，LM Listener单独计算。

每页包括当前训练/权重/恢复、官方推理入口和输入参数、数据缓存和适配、结果与待办。未取得完整资产的方法也保留独立页并明确缺项。

| 方法 | 当前状态 |
|---|---|
| [Learning2Listen（L2L，CVPR 2022）](learning2listen.md) | 四套官方人物权重各43条动作及GPAvatar视频已完成 |
| [ListenFormer](listenformer.md) | 本地完整200轮ViCo重训；43条动作、视频与统一指标完成 |
| [DyStream](dystream.md) | 官方权重43条视频与统一指标完成 |
| [Avatar Forcing](avatarforcing.md) | 官方权重43条结果；统一动作与五项图像重测完成 |
| [DualTalk](dualtalk.md) | 官方权重43条原生动作完成；RGB和统一EMOCA待接入 |
| [GLARE](glare.md) | 当前训练记录已推进至v9；自训练权重短片推理通过，正式ViCo43待测 |
| [ViCo baseline / RLHG](vico.md) | 仅诊断训练；正式重训在飞书主表暂缓 |
| [DIM（Dyadic Interaction Modeling）](dim.md) | 核心生成权重未就绪；正式重训暂缓 |
| [PCH / MM2022 ViCo Perceptual Head Generation](mm2022.md) | 代码集成和帮助入口通过；缺核心driver |
| [ReactFace](reactface.md) | 输入预处理和辅助模型通过；核心模型/渲染资源缺项 |
| [LM Listener（ICCV 2023）](lmlistener.md) | 两份核心预训练权重本地已存在；原生生成接入待完成 |
| [OmniResponse](omniresponse.md) | 数据/评测接入已有；未验证完整生成器和权重 |
| [SpeakerVid-5M](speakervid.md) | 数据整理/指标工具可用；无已接入对话生成器 |
| [DiTaiListener](ditailistener.md) | 待取得并接入官方生成资产 |
| [DiffListener](difflistener.md) | 待取得并接入官方生成资产 |
| [ELP](elp.md) | 待取得并接入官方生成资产 |
| [MFR-Net](mfrnet.md) | 待取得并接入官方生成资产 |
| [CustomListener](customlistener.md) | 待取得并接入官方生成资产 |
| [ARIG](arig.md) | 待取得并接入官方生成资产 |
| [ECHO](echo.md) | 待取得并接入官方生成资产 |
| [Latent Behavior Diffusion](latent_behavior_diffusion.md) | 待取得并接入官方生成资产 |

共同评测结果见[已完成43条评测及比较表](<../../experiments/p2_vico_full_eval_20261111/README.md>)；本轮AF追加结果见[AF评测及数值表](<../../experiments/p2_vico_avatarforcing_added_20261111/README.md>)，历史封存表保持不变。图像、BFM/DECA/EMOCA/SPECTRE各口径分别报告，不能仅凭同名FD认定可比。

GLARE：用户指定的v8恢复记录与当前文档顶部v9运行记录分别链接于其独立页；没有据此回退或重启训练。LM Listener：本次核实两份核心权重已在本地，状态更新为待接入生成，不能继续标记缺权重。

飞书Writing中的额外相关工作不自动作为主表独立待测方法；此索引覆盖[主表清单](<../../experiments/p2_vico_full_eval_20261111/feishu_scope/METHODS.csv>)及本次新增AF。资产可用性依据本地快照和已完成运行，不对缺失方法承诺推理已优化或已测完。
